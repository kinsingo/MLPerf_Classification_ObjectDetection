def define_env(env):

    @env.macro
    def mlperf_inference_implementation_readme(spaces, model, implementation):
        pre_space = ""

        for i in range(1,spaces):
            pre_space  = pre_space + " "
        f_pre_space = pre_space
        pre_space += " "

        content=""
        scenarios = []
        execution_envs = ["Docker","Native"]
        code_version="r4.1-dev"

        if model == "rnnt":
            code_version="r4.0"

        if implementation == "reference":
            # Tip
            if "99.9" not in model:
                content += f"\n{pre_space}!!! tip\n\n"
                content += f"{pre_space}    - MLCommons reference implementations are only meant to provide a rules compliant reference implementation for the submitters and in most cases are not best performing. If you want to benchmark any system, it is advisable to use the vendor MLPerf implementation for that system like Nvidia, Intel etc.\n\n"
              
            devices = [ "CPU", "CUDA", "ROCm" ]
            if model.lower() == "resnet50":
                 frameworks = [ "Onnxruntime", "Tensorflow", "Deepsparse" ]
            elif model.lower() == "retinanet":
                 frameworks = [ "Onnxruntime", "Pytorch" ]
            elif "bert" in model.lower():
                 frameworks = [ "Pytorch", "Deepsparse" ]
            else:
                 frameworks = [ "Pytorch" ]

        elif implementation == "nvidia":
            if model in [ "mixtral-8x7b" ]:
                 return pre_space+"    WIP"
            devices = [ "CUDA" ]
            frameworks = [ "TensorRT" ]
        
        elif implementation == "neuralmagic":
            devices = [ "CUDA" ]
            frameworks = [ "pytorch" ]

        elif implementation == "intel":
            # Tip
            if "99.9" not in model:
                content += f"\n{pre_space}!!! tip\n\n"
                content += f"{pre_space}    - Intel MLPerf inference implementation is available only for datacenter category and has been tested only on a limited number of systems. Most of the benchmarks using Intel implementation require at least Intel Sapphire Rapids or higher CPU generation.\n\n"
                                
            if model not in [ "bert-99", "bert-99.9", "gptj-99", "gptj-99.9", "resnet50", "retinanet", "3d-unet-99", "3d-unet-99.9", "dlrm-v2-99", "dlrm-v2-99.9", "sdxl" ]:
                 return pre_space+"    WIP"
            if model in [ "bert-99", "bert-99.9", "retinanet", "3d-unet-99", "3d-unet-99.9" ]:
                 code_version="r4.0"
            devices = [ "CPU" ]
            frameworks = [ "Pytorch" ]

        elif implementation == "qualcomm":
            if model not in [ "resnet50", "retinanet", "bert-99", "bert-99.9" ]:
                 return pre_space+"    WIP"

            devices = [ "QAIC" ]
            frameworks = [ "Glow" ]

        elif implementation == "cpp":
            devices = [ "CPU", "CUDA" ]
            frameworks = [ "Onnxruntime" ]

        elif implementation == "ctuning-cpp":
            scenarios = [ "SingleStream" ]
            devices = [ "CPU" ]
            if model.lower() == "resnet50":
                 frameworks = [ "TFLite" ]
            else:
                 frameworks = []

        if model.lower() == "bert-99.9":
            categories = [ "Datacenter" ]
        elif "dlrm" in model.lower() or "llama2" in model.lower() or "mixtral" in model.lower():
            categories = [ "Datacenter" ]
        else:
            categories = [ "Edge", "Datacenter" ]

        # model name
        content += f"{pre_space}{model.upper()}\n\n"
        for category in categories:
            if category == "Edge" and not scenarios:
                scenarios = [ "Offline", "SingleStream" ]
                if model.lower() in [ "resnet50", "retinanet" ] and not "MultiStream" in scenarios:#MultiStream was duplicating
                     scenarios.append("MultiStream")
            elif category == "Datacenter":
                 scenarios = [ "Offline", "Server" ] 

            content += f"{pre_space}=== \"{category.lower()}\"\n\n"

            cur_space = pre_space + "    "
            scenarios_string = ", ".join(scenarios)

            content += f"{cur_space}### {category} category \n\n{cur_space} In the {category.lower()} category, {model} has {scenarios_string} scenarios and all the scenarios are mandatory for a closed division submission.\n\n"


            for framework in frameworks:
                cur_space1 = cur_space + "    "
                content += f"{cur_space}=== \"{framework}\"\n"
                content += f"{cur_space1}#### {framework} framework\n\n"

                for device in devices:
                    if framework.lower() == "deepsparse":
                        if device.lower() != "cpu":
                             continue
                    cur_space2 = cur_space1 + "    "
                    cur_space3 = cur_space2 + "    "
                    cur_space4 = cur_space3 + "    "
                    
                    content += f"{cur_space1}=== \"{device}\"\n"
                    content += f"{cur_space2}##### {device} device\n\n"

                    # minimum system requirements
                    content += get_min_system_requirements(cur_space2, model, implementation, device)

                    # to select the execution environments(currently Docker and Native)
                    for execution_env in execution_envs:
                        if (device == "ROCm" or implementation == "qualcomm") and execution_env == "Docker":
                            continue  # docker not currently supported for Qualcomm implementation and ROCm device
                        if implementation == "nvidia" and execution_env == "Native":
                            continue  # Nvidia implementation only supports execution through docker
                        content += f"{cur_space2}=== \"{execution_env}\"\n"
                        content += f"{cur_space3}###### {execution_env} Environment\n\n"
                        # ref to cm installation
                        content += f"{cur_space3}Please refer to the [installation page](../../install/index.md) to install CM for running the automated benchmark commands.\n\n"
                        test_query_count=get_test_query_count(model, implementation, device)

                        if "99.9" not in model: #not showing docker command as it is already done for the 99% variant
                            if implementation == "neuralmagic":
                                content += f"{cur_space3}####### Run the Inference Server\n"
                                content += get_inference_server_run_cmd(spaces+16,implementation)
                                # tips regarding the running of nural magic server
                                content += f"\n{cur_space3}!!! tip\n\n"
                                content += f"{cur_space3}    - Host and Port number of the server can be configured through `--host` and `--port`. Otherwise, server will run on default host `localhost` and port `8000`.\n\n"
                                
                            if execution_env == "Native": # Native implementation steps through virtual environment
                                content += f"{cur_space3}####### Setup a virtual environment for Python\n"
                                content += get_venv_command(spaces+16)
                                content += f"{cur_space3}####### Performance Estimation for Offline Scenario\n"
                                content += mlperf_inference_run_command(spaces+17, model, implementation, framework.lower(), category.lower(), "Offline", device.lower(), "test", test_query_count, True, scenarios, code_version).replace("--docker ","")
                                content += f"{cur_space3}The above command should do a test run of Offline scenario and record the estimated offline_target_qps.\n\n"

                            else: # Docker implementation steps
                                content += f"{cur_space3}####### Docker Container Build and Performance Estimation for Offline Scenario\n"
                                docker_info = get_docker_info(spaces+16, model, implementation, device)
                                content += docker_info
                                content += mlperf_inference_run_command(spaces+17, model, implementation, framework.lower(), category.lower(), "Offline", device.lower(), "test", test_query_count, True, scenarios, code_version)
                                content += f"{cur_space3}The above command should get you to an interactive shell inside the docker container and do a quick test run for the Offline scenario. Once inside the docker container please do the below commands to do the accuracy + performance runs for each scenario.\n\n"
                                content += f"{cur_space3}<details>\n"
                                content += f"{cur_space3}<summary> Please click here to see more options for the docker launch </summary>\n\n"
                                content += f"{cur_space3}* `--docker_cm_repo=<Custom CM repo URL>`: to use a custom fork of cm4mlops repository inside the docker image\n\n"
                                content += f"{cur_space3}* `--docker_cache=no`: to not use docker cache during the image build\n"

                                if device.lower() not in [ "cuda" ]:
                                    content += f"{cur_space3}* `--docker_os=ubuntu`: ubuntu and rhel are supported. \n"
                                    content += f"{cur_space3}* `--docker_os_version=20.04`: [20.04, 22.04] are supported for Ubuntu and [8, 9] for RHEL\n"

                                content += f"{cur_space3}</details>\n"
                        else:
                            content += f"{cur_space3} You can reuse the same environment as described for {model.split('.')[0]}.\n"
                            content += f"{cur_space3}###### Performance Estimation for Offline Scenario\n"
                            content += mlperf_inference_run_command(spaces+17, model, implementation, framework.lower(), category.lower(), "Offline", device.lower(), "test", test_query_count, True, scenarios, code_version).replace("--docker ","")
                            content += f"{cur_space3}The above command should do a test run of Offline scenario and record the estimated offline_target_qps.\n\n"


                        run_suffix = ""
                        run_suffix += f"{cur_space3}<details>\n"
                        run_suffix += f"{cur_space3}<summary> Please click here to see more options for the RUN command</summary>\n\n"
                        run_suffix += f"{cur_space3}* Use `--division=closed` to do a closed division submission which includes compliance runs\n\n"
                        run_suffix += f"{cur_space3}* Use `--rerun` to do a rerun even when a valid run exists\n"  
                        run_suffix += f"{cur_space3}</details>\n\n"

                        if "bert" in model.lower() and framework == "deepsparse":
                            run_suffix += f"{cur_space3}<details>\n"
                            run_suffix += f"{cur_space3}<summary> Please click here for generic model stubs for bert deepsparse</summary>\n\n"
                            run_suffix += f"{cur_space3}* zoo:nlp/question_answering/obert-large/pytorch/huggingface/squad/pruned95_quant-none-vnni\n\n" 
                            run_suffix += f"{cur_space3}* zoo:nlp/question_answering/mobilebert-none/pytorch/huggingface/squad/14layer_pruned50_quant-none-vnni\n\n" 
                            run_suffix += f"{cur_space3}* zoo:nlp/question_answering/mobilebert-none/pytorch/huggingface/squad/base_quant-none\n\n"
                            run_suffix += f"{cur_space3}* zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned95_obs_quant-none\n\n"
                            run_suffix += f"{cur_space3}* zoo:nlp/question_answering/mobilebert-none/pytorch/huggingface/squad/14layer_pruned50-none-vnni\n\n"
                            run_suffix += f"{cur_space3}* zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/pruned90-none\n\n"
                            run_suffix += f"{cur_space3}* zoo:nlp/question_answering/obert-large/pytorch/huggingface/squad/pruned97_quant-none\n\n"
                            run_suffix += f"{cur_space3}* zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned90-none\n\n"
                            run_suffix += f"{cur_space3}* zoo:nlp/question_answering/bert-large/pytorch/huggingface/squad/pruned80_quant-none-vnni\n\n"
                            run_suffix += f"{cur_space3}* zoo:nlp/question_answering/obert-large/pytorch/huggingface/squad/pruned95-none-vnni\n\n"
                            run_suffix += f"{cur_space3}* zoo:nlp/question_answering/obert-large/pytorch/huggingface/squad/pruned97-none\n\n"
                            run_suffix += f"{cur_space3}* zoo:nlp/question_answering/bert-large/pytorch/huggingface/squad/base-none\n\n"
                            run_suffix += f"{cur_space3}* zoo:nlp/question_answering/obert-large/pytorch/huggingface/squad/base-none\n\n"
                            run_suffix += f"{cur_space3}* zoo:nlp/question_answering/mobilebert-none/pytorch/huggingface/squad/base-none\n"
                            run_suffix += f"{cur_space3}</details>\n"

                        

                        for scenario in scenarios:
                            content += f"{cur_space3}=== \"{scenario}\"\n{cur_space4}###### {scenario}\n\n"
                            run_cmd = mlperf_inference_run_command(spaces+21, model, implementation, framework.lower(), category.lower(), scenario, device.lower(), "valid", 0, False, scenarios, code_version)
                            content += run_cmd
                            #content += run_suffix
 
                        content += f"{cur_space3}=== \"All Scenarios\"\n{cur_space4}###### All Scenarios\n\n"
                        run_cmd = mlperf_inference_run_command(spaces+21, model, implementation, framework.lower(), category.lower(), "All Scenarios", device.lower(), "valid", 0, False, scenarios, code_version)
                        content += run_cmd
                        content += run_suffix

                    

        readme_prefix = get_readme_prefix(spaces, model, implementation)

        readme_suffix = get_readme_suffix(spaces, model, implementation)

        return readme_prefix + content + readme_suffix

    def get_test_query_count(model, implementation, device, num_devices=1):

        if model == "resnet50":
             p_range = 1000
        elif model in [ "retinanet", "bert-99", "bert-99.9" ]:
             p_range = 100
        else:
             p_range = 50

        if device == "cuda":
            p_range *= 40
            p_range *= num_devices

        return p_range
    
    def get_min_system_requirements(spaces, model, implementation, device):
        model = model.lower()
        min_sys_req_content = ""
        min_sys_req_content += f"{spaces}<details>\n"
        min_sys_req_content += f"{spaces}<summary>Please click here to see the minimum system requirements for running the benchmark</summary>\n\n"
        # device memory
        if device.lower() == "cuda" and (implementation.lower() == "nvidia" or implementation.lower() == "reference"):
            if implementation.lower() == "nvidia":
                if "dlrm" in model:
                    device_memory = "24GB"
                elif "llama2-70b" in model or "mixtral" in model:
                    device_memory = "80GB"
                elif "sdxl" in model or "gptj" in model:
                    device_memory = "16GB"
                else:
                    device_memory = "8GB"
            elif implementation.lower() == "reference":
                if "dlrm" in model:
                    device_memory = "2x80GB"
                elif "llama2-70b" in model:
                    device_memory = "8x80GB"
                elif "mixtral" in model:
                    device_memory = "4x80GB"
                elif "sdxl" in model:
                    device_memory = "24GB(fp32), 16GB(fp16)"
                elif "gptj" in model:
                    device_memory = "80GB(fp32). 40GB(fp16)"
                else:
                    device_memory = "8GB"
            min_sys_req_content += f"{spaces}* **Device Memory**: {device_memory}\n\n"
        # disk space
        if "dlrm" in model:
            disk_space = "500GB"
        elif "llama2-70b" in model:
            disk_space = "700GB"
        elif "mixtral" in model:
            disk_space = "100GB"
        elif "retinanet" in model:
            disk_space = "200GB"
        else:
            disk_space = "50GB"
        min_sys_req_content += f"{spaces}* **Disk Space**: {disk_space}\n\n"
        # System memory
        if "dlrm" in model:
            system_memory = "512GB"
            min_sys_req_content += f"{spaces}* **System Memory(RAM+SWAP)**: {system_memory}\n\n"
        min_sys_req_content += f"{spaces}</details>\n"
        return min_sys_req_content

    def get_readme_prefix(spaces, model, implementation):
        readme_prefix = ""
        pre_space="    "
        #for i in range(1,spaces):
        #     pre_space  = pre_space + " "
        #pre_space += "  "

        return readme_prefix
    
    def get_inference_server_run_cmd(spaces, implementation):
        indent = " "*spaces + " "
        if implementation == "neuralmagic":
            pre_space = " "*spaces
            return f"""\n
{pre_space}```bash
{pre_space}cm run script --tags=run,vllm-server \\
{indent}--model=nm-testing/Llama-2-70b-chat-hf-FP8 \\
{indent}--vllm_model_name=nm-testing/Llama-2-70b-chat-hf-FP8 \\
{indent}--quiet
{pre_space}```\n"""

    def get_venv_command(spaces):
      pre_space = " "*spaces
      return f"""\n
{pre_space}```bash
{pre_space}cm run script --tags=install,python-venv --name=mlperf
{pre_space}export CM_SCRIPT_EXTRA_CMD=\"--adr.python.name=mlperf\"
{pre_space}```\n"""   

    def get_docker_info(spaces, model, implementation, device):
        info = ""
        pre_space=""
        for i in range(1,spaces):
             pre_space  = pre_space + " "
        pre_space += " "
        #pre_space = "                "
        if implementation == "nvidia":
            info += f"\n{pre_space}!!! tip\n\n"
            info+= f"{pre_space}    If ran with `--all_models=yes`, all the benchmark models of NVIDIA implementation could be run within the same container.\n\n"
        return info

    def get_readme_suffix(spaces, model, implementation):
        readme_suffix = ""
        pre_space=""
        for i in range(1,spaces):
             pre_space  = pre_space + " "
        pre_space += "  "

        if implementation == "reference":
            if not model.endswith("-99"):
                model_base_name = model.replace("-99.9","").replace("-99","")
                readme_suffix+= f"{pre_space}* If you want to download the official MLPerf model and dataset for {model} you can follow [this README](get-{model_base_name}-data.md).\n"
            if model == "resnet50":
                 readme_suffix += f"{pre_space}* Please see [mobilenets.md](mobilenets.md) for running mobilenet models for Image Classification."
        return readme_suffix

    def get_run_cmd_extra(f_pre_space, model, implementation, device, scenario, scenarios = []):
        extra_content = ""
        f_pre_space += ""
        if scenario == "Server" or (scenario == "All Scenarios" and "Server" in scenarios):
            extra_content += f"{f_pre_space}    * `<SERVER_TARGET_QPS>` must be determined manually. It is usually around 80% of the Offline QPS, but on some systems, it can drop below 50%. If a higher value is specified, the latency constraint will not be met, and the run will be considered invalid.\n"

        if "gptj" in model and device == "cuda" and implementation == "reference":
            extra_content += f"{f_pre_space}    * `--precision=[float16|bfloat16]` can help run on GPUs with less RAM \n"
            extra_content += f"{f_pre_space}    * `--beam-size=1` Beam size of 4 is mandatory for a closed division submission but reducing the beam size can help in running the model on GPUs with lower device memory\n"
        if extra_content:
            extra_content = f"{f_pre_space}!!! tip\n\n" + extra_content

        return extra_content

    @env.macro
    def mlperf_inference_run_command(spaces, model, implementation, framework, category, scenario, device="cpu", execution_mode="test", test_query_count="20", docker=False, scenarios = [], code_version="r4.1-dev"):
        pre_space = ""
        for i in range(1,spaces):
             pre_space  = pre_space + " "
        f_pre_space = pre_space
        pre_space += "  "

        if scenario == "All Scenarios":
             scenario_variation_tag = ",_all-scenarios"
             scenario_option = ""
        else:
            scenario_variation_tag = ""
            scenario_option = f"\\\n{pre_space} --scenario={scenario}"

        if scenario == "Server" or (scenario == "All Scenarios" and "Server" in scenarios):
            scenario_option += f"\\\n{pre_space} --server_target_qps=<SERVER_TARGET_QPS>"

        run_cmd_extra = get_run_cmd_extra(f_pre_space, model, implementation, device, scenario, scenarios)

        if docker:
            docker_cmd_suffix = f" \\\n{pre_space} --docker --quiet"
            docker_cmd_suffix += f" \\\n{pre_space} --test_query_count={test_query_count}"
            
            if "bert" in model.lower() and framework == "deepsparse":
                docker_cmd_suffix += f"\\\n{pre_space} --env.CM_MLPERF_NEURALMAGIC_MODEL_ZOO_STUB=zoo:nlp/question_answering/mobilebert-none/pytorch/huggingface/squad/base_quant-none"
            if "llama2-70b" in model.lower():
                if implementation == "nvidia":
                    docker_cmd_suffix += f" \\\n{pre_space} --tp_size=2"
                    docker_cmd_suffix += f" \\\n{pre_space} --nvidia_llama2_dataset_file_path=<PATH_TO_PICKE_FILE>"
                elif implementation == "neuralmagic":
                    docker_cmd_suffix += f" \\\n{pre_space} --api_server=http://localhost:8000"
                    docker_cmd_suffix += f" \\\n{pre_space} --vllm_model_name=nm-testing/Llama-2-70b-chat-hf-FP8"
                    docker_cmd_suffix += f" \\\n{pre_space} --adr.mlperf-implementation.tags=_repo.https://github.com/neuralmagic/inference,_branch.vllm"
            
            if "dlrm-v2" in model.lower() and implementation == "nvidia":
                docker_cmd_suffix += f" \\\n{pre_space} --criteo_day23_raw_data_path=<PATH_TO_CRITEO_DAY23_RAW_DATA>"

            docker_setup_cmd = f"""\n
{f_pre_space}```bash
{f_pre_space}cm run script --tags=run-mlperf,inference,_find-performance,_full,_{code_version}{scenario_variation_tag} \\
{pre_space} --model={model} \\
{pre_space} --implementation={implementation} \\
{pre_space} --framework={framework} \\
{pre_space} --category={category} {scenario_option} \\
{pre_space} --execution_mode=test \\
{pre_space} --device={device} {docker_cmd_suffix}
{f_pre_space}```\n"""

            return docker_setup_cmd + run_cmd_extra

        else:
            cmd_suffix = f"\\\n{pre_space} --quiet"

            if execution_mode == "test":
                cmd_suffix += f" \\\n {pre_space} --test_query_count={test_query_count}"

            if "bert" in model.lower() and framework == "deepsparse":
                cmd_suffix += f"\\\n{pre_space} --env.CM_MLPERF_NEURALMAGIC_MODEL_ZOO_STUB=zoo:nlp/question_answering/mobilebert-none/pytorch/huggingface/squad/base_quant-none"
            if "llama2-70b" in model.lower():
                if implementation == "nvidia":
                    cmd_suffix += f" \\\n{pre_space} --tp_size=<TP_SIZE>"
                    cmd_suffix += f" \\\n{pre_space} --nvidia_llama2_dataset_file_path=<PATH_TO_PICKE_FILE>"
                elif implementation == "neuralmagic":
                    cmd_suffix += f" \\\n{pre_space} --api_server=http://localhost:8000"
                    cmd_suffix += f" \\\n{pre_space} --vllm_model_name=nm-testing/Llama-2-70b-chat-hf-FP8"
                    cmd_suffix += f" \\\n{pre_space} --adr.mlperf-implementation.tags=_repo.https://github.com/neuralmagic/inference,_branch.vllm"
            
            if "dlrm-v2" in model and implementation == "nvidia":
                cmd_suffix += f" \\\n{pre_space} --criteo_day23_raw_data_path=<PATH_TO_CRITEO_DAY23_RAW_DATA>"

            run_cmd = f"""\n
{f_pre_space}```bash
{f_pre_space}cm run script --tags=run-mlperf,inference,_{code_version}{scenario_variation_tag} \\
{pre_space} --model={model} \\
{pre_space} --implementation={implementation} \\
{pre_space} --framework={framework} \\
{pre_space} --category={category} {scenario_option} \\
{pre_space} --execution_mode={execution_mode} \\
{pre_space} --device={device} {cmd_suffix}
{f_pre_space}```\n"""

            return run_cmd + run_cmd_extra