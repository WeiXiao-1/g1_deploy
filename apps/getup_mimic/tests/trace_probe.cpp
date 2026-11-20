// Offline trace queue/file probe. No SDK, DDS or motor publication.
#include "g1_getup_mimic_trace.h"
#include <cassert>
#include <csignal>
#include <iostream>
#include <sys/resource.h>
using namespace g1_getup_mimic::trace;

int main(int argc,char** argv) {
    if(argc==2 && std::string(argv[1])=="queue") {
        Queue<ControlRecord,2> queue;
        for(std::uint64_t id=1;id<=2;++id) {auto* p=queue.Begin();assert(p);p->id=id;queue.Commit();}
        assert(!queue.Begin());
        ControlRecord record;
        assert(queue.Pop(record)&&record.id==1);
        auto* p=queue.Begin();assert(p&&p->id==0);p->id=3;queue.Commit();
        assert(queue.Pop(record)&&record.id==2);
        assert(queue.Pop(record)&&record.id==3);
        assert(!queue.Pop(record));
        Queue<ControlRecord,128> concurrent;
        std::thread producer([&]{for(std::uint64_t i=1;i<=20000;++i){ControlRecord* p;while(!(p=concurrent.Begin()))std::this_thread::yield();p->id=i;concurrent.Commit();}});
        for(std::uint64_t i=1;i<=20000;++i){while(!concurrent.Pop(record))std::this_thread::yield();assert(record.id==i);}
        producer.join();std::cout<<"queue ok\n";return 0;
    }
    if(argc!=5)return 2;
    Metadata meta;
    meta.config_path=argv[2];meta.reset_path=argv[2];meta.model_path=argv[2];
    meta.reset_pose="dwaq-reset";meta.actuation="drive_pd";
    for(int i=0;i<29;++i){meta.policy_names.push_back("p"+std::to_string(i));meta.motor_names.push_back("m"+std::to_string(i));}
    Recorder recorder(argv[1],meta);
    const bool failure=std::string(argv[3])=="failure";
    if(failure){std::signal(SIGXFSZ,SIG_IGN);rlimit limit{16384,16384};if(setrlimit(RLIMIT_FSIZE,&limit))return 3;}
    const int controls=std::stoi(argv[4]);
    std::thread inference([&]{
        for(int i=1;i<=100;++i)if(auto* p=recorder.BeginInference()) {
            p->id=i;p->activation_id=1;p->flags=1;p->control_sample_id=i;
            p->producer_begin_ns=NowNs();p->submitted_ns=p->state_received_ns=p->producer_begin_ns-100;
            p->inference_start_ns=NowNs();p->policy.observation_size=744;
            for(int j=0;j<744;++j)p->policy.observation[j]=float(i+j)/1000;
            p->policy.onnx_action[3]=float(i);p->policy.applied_action[3]=float(i);
            p->target[4]=-.2f;p->q[4]=-.21f;p->kp[4]=40;
            p->inference_end_ns=NowNs();p->enqueue_ready_ns=NowNs();recorder.CommitInference();
        }
    });
    for(int i=1;i<=controls;++i) {
        ControlScope scope(&recorder,i);
        if(auto* p=scope.Get()) {
            p->activation_id=1;p->phase=Policy;p->flags=StateValid|CommandPrepared;
            p->q[4]=-.21f;p->command_q[4]=-.2f;p->command_kp[4]=40;
            p->effective_kp[4]=40;p->estimated_pd_torque[4]=.4f;
        }
    }
    inference.join();recorder.Close();
    std::cout<<recorder.Directory().string()<<'\n';
    return failure && !recorder.Error()?4:0;
}
