// Verify optional trace capture against actual ONNX inference. No DDS or motors.
#include "g1_getup_mimic_policy.h"
#include "g1_config_utils.h"
#include "g1_joint_mapping.h"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace g1_getup_mimic;
int main(int argc,char** argv) {
    if(argc!=4)return 2;
    PolicyConfig config(argv[1],argv[2]);
    Policy plain(config),logged(config);
    const auto names=config.root["joint_names"].as<std::vector<std::string>>();
    const auto ids=g1_joints::FromNames(names);
    const auto nominal=g1_config::ReadArray<float,kJointCount>(config.root["default_joint_pos"]);
    const auto scale=g1_config::ReadArray<float,kJointCount>(config.root["action_scale"]);
    trace::Metadata metadata;
    metadata.config_path=config.path;
    metadata.reset_path=g1_config::ResolvePath(config.path,config.root["reset_path"].as<std::string>());
    metadata.model_path=g1_config::ResolvePath(config.path,config.model["policy_path"].as<std::string>());
    metadata.policy_names=names;metadata.motor_names.assign(g1_joints::kDdsNames.begin(),g1_joints::kDdsNames.end());
    trace::Recorder recorder(argv[3],metadata);
    for(int i=0;i<24;++i) {
        State state;
        state.position=g1_joints::ToMotorOrder(nominal,ids);
        state.position[0]+=.02f*std::sin(float(i)*.1f);
        const auto expected=plain.Step(state);
        auto* record=recorder.BeginInference();assert(record);
        record->id=i+1;record->flags=1;record->producer_begin_ns=trace::NowNs();
        record->inference_start_ns=trace::NowNs();
        const auto actual=logged.Step(state,&record->policy);
        record->inference_end_ns=trace::NowNs();
        assert(expected.position==actual.position && expected.kp==actual.kp && expected.kd==actual.kd);
        const auto predicted=logged.Predict(record->policy.observation.data(),record->policy.observation_size);
        for(int j=0;j<kJointCount;++j) {
            assert(std::abs(predicted[j]-record->policy.onnx_action[j])<1e-5f);
            assert(actual.position[ids[j]]==nominal[j]+scale[j]*record->policy.applied_action[j]);
        }
        if(i==0) {
            if(config.observation_size==kObservationSize) {
                for(int h=0;h<8;++h)assert(record->policy.observation[24+3*h+2]==-1.f);
                for(int j=512;j<744;++j)assert(record->policy.observation[j]==0.f);
            }else assert(record->policy.observation[5]==-1.f);
        }
        record->target=actual.position;record->kp=actual.kp;record->kd=actual.kd;
        record->enqueue_ready_ns=trace::NowNs();recorder.CommitInference();
    }
    recorder.Close();std::cout<<recorder.Directory().string()<<'\n';
}
