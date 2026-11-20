"""Trace writer/decoder tests using a standalone C++ probe; never connects to DDS."""
import importlib.util
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3]
SPEC=importlib.util.spec_from_file_location('getup_read_trace',ROOT/'apps/getup_mimic/tools/read_trace.py')
reader=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


@unittest.skipUnless(shutil.which('g++'), 'g++ required for offline trace probe')
class TraceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='g1-trace-test-')
        cls.base=Path(cls.temp.name)
        cls.probe=cls.base/'probe'
        cls.config=cls.base/'input.json';cls.config.write_text('{"test":true}\n')
        subprocess.run(['g++','-std=c++17','-O2','-pthread','-Wall','-Wextra','-Werror',
                        '-I'+str(ROOT/'apps/getup_mimic/include'),
                        str(ROOT/'apps/getup_mimic/tests/trace_probe.cpp'),'-o',str(cls.probe)],check=True,capture_output=True,text=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def make_trace(self,mode='normal',count=500):
        parent=Path(tempfile.mkdtemp(dir=self.base))
        result=subprocess.run([str(self.probe),str(parent),str(self.config),mode,str(count)],check=True,capture_output=True,text=True,timeout=20)
        return Path(result.stdout.strip())

    def test_queue_capacity_fifo_wraparound_and_threads(self):
        result=subprocess.run([str(self.probe),'queue'],capture_output=True,text=True,check=True,timeout=10)
        self.assertIn('queue ok',result.stdout)

    def test_binary_roundtrip_and_abi_manifest(self):
        run=self.make_trace()
        warnings=[];rows=list(reader.records(run,warnings));self.assertFalse(warnings)
        inf=[r for r in rows if r['type']=='inference'];ctrl=[r for r in rows if r['type']=='control']
        self.assertEqual(len(inf),100);self.assertEqual(len(ctrl),500)
        self.assertEqual([r['id'] for r in inf],list(range(1,101)))
        for r in inf:
            self.assertEqual(len(r['observation']),744)
            self.assertAlmostEqual(r['observation'][743],(r['id']+743)/1000,places=6)
            self.assertEqual(r['onnx_action'][3],r['id'])
        self.assertAlmostEqual(ctrl[0]['command_q'][4],-.2,places=6)
        summary=reader.decode(run,run/'decoded')
        self.assertEqual(summary['decoded_records'],{'inference':100,'control':500})
        self.assertEqual(summary['final_recorder_stats']['write_error_errno'],0)
        self.assertFalse(summary['warnings'])
        self.assertEqual(summary['timing']['active_control_work']['count'],500)

    def test_overload_counts_drops_and_preserves_order(self):
        run=self.make_trace(count=50000)
        rows=list(reader.records(run,[]));stats=json.loads((run/'stats.json').read_text())
        for kind,attempts in [('inference',100),('control',50000)]:
            ids=[r['id'] for r in rows if r['type']==kind]
            self.assertEqual(ids,sorted(set(ids)))
            self.assertEqual(len(ids),stats[kind+'_written'])
            self.assertEqual(len(ids)+stats[kind+'_dropped'],attempts)

    def test_writer_failure_is_nonfatal_and_recoverable(self):
        run=self.make_trace(mode='failure',count=50000)
        stats=json.loads((run/'stats.json').read_text())
        self.assertNotEqual(stats['write_error_errno'],0)
        summary=reader.decode(run,run/'decoded')
        self.assertTrue(summary['warnings'])
        self.assertGreater(summary['decoded_records']['control'],0)

    def test_incomplete_tail_recovers_complete_records(self):
        run=self.make_trace(count=20)
        path=run/'records.bin';data=path.read_bytes();path.write_bytes(data[:-11])
        summary=reader.decode(run,run/'decoded')
        self.assertTrue(any('Incomplete final' in warning for warning in summary['warnings']))


@unittest.skipUnless((ROOT/'build/CMakeFiles/g1_getup_mimic.dir/apps/getup_mimic/src/policy.cpp.o').exists()
                     and shutil.which('g++'), 'Build g1_getup_mimic before ONNX trace tests')
class PolicyTraceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='g1-policy-trace-test-')
        cls.base=Path(cls.temp.name);cls.probe=cls.base/'probe'
        cmake=ROOT/'build/CMakeFiles/g1_getup_mimic.dir'
        flags=(cmake/'flags.make').read_text().splitlines()
        includes=shlex.split(next(s.split('=',1)[1] for s in flags if s.startswith('CXX_INCLUDES =')))
        link=shlex.split((cmake/'link.txt').read_text())
        libraries=[s for s in link if s.startswith('-Wl,') or '.so' in s]
        subprocess.run(['g++','-std=c++17','-O2','-pthread',*includes,
                        str(ROOT/'apps/getup_mimic/tests/trace_policy_probe.cpp'),
                        str(cmake/'apps/getup_mimic/src/policy.cpp.o'),*libraries,'-o',str(cls.probe)],
                       capture_output=True,text=True,check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_captured_inputs_reproduce_onnx_outputs_without_changing_targets(self):
        for expert,size in [('',744),('faceup',93),('backup',93)]:
            with self.subTest(expert=expert or 'model_14000'):
                result=subprocess.run([str(self.probe),str(ROOT/'apps/getup_mimic/config/g1_getup_mimic.json'),
                                       expert,str(self.base)],capture_output=True,text=True,check=True,timeout=20)
                run=Path(result.stdout.strip());rows=list(reader.records(run,[]))
                self.assertEqual(len(rows),24)
                self.assertTrue(all(len(r['observation'])==size and r['flags']==1 for r in rows))


if __name__=='__main__':
    unittest.main()
