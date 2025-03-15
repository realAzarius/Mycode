import os
import sys
import argparse
import numpy as np
import random
import time
from datetime import datetime
import copy
from sys import getsizeof
import sqlite3
import pickle
from pathlib import Path
import shutil
import torch
import torch.nn.functional as F
from Models import simplecnn, Mnist_CNN, Mnist_2NN, Mnist_CNN2
from Device import Device, DevicesInNetwork
from Block import Block
import glog as log
from Blockchain import Blockchain
import math
from configparser import ConfigParser
from sklearn.decomposition import PCA
from Device import update_graph_matrix_neighbor
from collections import defaultdict

from phe.encoding import EncodedNumber

import phe.paillier
# 生成Gram矩阵
from phe import paillier

'''
项目的主文件
'''

# set program execution time for logging purpose
date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_files_folder_path = f"logs/{date_time}"
NETWORK_SNAPSHOTS_BASE_FOLDER = "snapshots"

if __name__ == "__main__":

    # create logs/ if not exists
    if not os.path.exists('logs'):
        os.makedirs('logs')

    args = ConfigParser()
    args.read('./config/config.ini', encoding='utf-8')

    config = args[args.sections()[0]]

    # debug attributes
    gpu = config.getint('gpu')
    verbose = config.getint('verbose')
    save_network_snapshots = config.getint('save_network_snapshots')
    destroy_tx_in_block = config.getint('destroy_tx_in_block')
    resume_path = config.get('resume_path')
    save_freq = config.getint('save_freq')
    save_most_recent = config.getint('save_most_recent')

    # FL attributes
    batch_size = config.getint('batch_size')
    model_name = config.get('model_name')
    learning_rate = config.getfloat('learning_rate')
    optimizer = config.get('optimizer')
    IID = config.getint('IID')
    max_num_comm = config.getint('max_num_comm')
    num_enterprise = config.getint('num_enterprise')
    shard_test_data = config.getint('shard_test_data')
    num_malicious = config.getint('num_malicious')
    noise_variance = config.getint('noise_variance')
    default_local_epochs = config.getint('default_local_epochs')

    # blockchain system consensus attributes
    unit_reward = config.getint('unit_reward')
    knock_out_rounds = config.getint('knock_out_rounds')
    lazy_local_enterprise_knock_out_rounds = config.getint('lazy_local_enterprise_knock_out_rounds')
    pow_difficulty = config.getint('pow_difficulty')

    # blockchain FL validator/miner restriction tuning parameters
    miner_acception_wait_time = config.getfloat('miner_acception_wait_time')
    miner_accepted_transactions_size_limit = config.getfloat('miner_accepted_transactions_size_limit')
    miner_pos_propagated_block_wait_time = config.getfloat('miner_pos_propagated_block_wait_time')
    validator_threshold = config.getfloat('validator_threshold')
    malicious_updates_discount = config.getfloat('malicious_updates_discount')
    malicious_validator_on = config.getint('malicious_validator_on')

    # distributed system attributes
    network_stability = config.getfloat('network_stability')
    even_link_speed_strength = config.getint('even_link_speed_strength')
    base_data_transmission_speed = config.getfloat('base_data_transmission_speed')
    even_computation_power = config.getint('even_computation_power')

    # simulation attributes
    hard_assign = config.get('hard_assign')
    all_in_one = config.getint('all_in_one')
    check_signature = config.getint('check_signature')
    attack_type = config.get('attack_type')
    target_acc = config.getfloat('target_acc')

    # new attribute
    k_principal = config.getint('k_principal')

    # detect CUDA
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # dev = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    # pre-define system variables
    latest_round_num = 0

    ''' If network_snapshot is specified, continue from left '''
    if resume_path != 'None':
        if not save_network_snapshots:
            print("NOTE: save_network_snapshots is set to 0. New network_snapshots won't be saved by conituing.")
        network_snapshot_save_path = f"{NETWORK_SNAPSHOTS_BASE_FOLDER}/{resume_path}"
        latest_network_snapshot_file_name = \
            sorted([f for f in os.listdir(network_snapshot_save_path) if not f.startswith('.')],
                   key=lambda fn: int(fn.split('_')[-1]), reverse=True)[0]
        print(f"Loading network snapshot from {resume_path}/{latest_network_snapshot_file_name}")
        print("BE CAREFUL - loaded dev env must be the same as the current dev env, namely, cpu, gpu or gpu parallel")
        latest_round_num = int(latest_network_snapshot_file_name.split('_')[-1])
        devices_in_network = pickle.load(
            open(f"{network_snapshot_save_path}/{latest_network_snapshot_file_name}", "rb"))
        devices_list = list(devices_in_network.devices_set.values())
        log_files_folder_path = f"/share/home/MP2209117/proj/logs/{resume_path}"
        # for colab
        # log_files_folder_path = f"/content/drive/MyDrive/BFA/logs/{args['resume_path']}"
        # original arguments file
        args_used_file = f"{log_files_folder_path}/args_used.txt"
        file = open(args_used_file, "r")
        log_whole_text = file.read()
        lines_list = log_whole_text.split("\n")
        for line in lines_list:
            # abide by the original specified rewards
            if line.startswith('--unit_reward'):
                rewards = int(line.split(" ")[-1])
            # get number of roles
            if line.startswith('--hard_assign'):
                roles_requirement = line.split(" ")[-1].split(',')
            # get mining consensus
            if line.startswith('--pow_difficulty'):
                mining_consensus = 'PoW' if int(line.split(" ")[-1]) else 'PoS'
        # determine roles to assign
        try:
            workers_needed = int(roles_requirement[0])
        except:
            workers_needed = 1
        try:
            validators_needed = int(roles_requirement[1])
        except:
            validators_needed = 1
        try:
            miners_needed = int(roles_requirement[2])
        except:
            miners_needed = 1
    else:
        ''' SETTING UP FROM SCRATCH'''

        # 0. create log_files_folder_path if not resume
        os.mkdir(log_files_folder_path)

        with open('./config/config.ini', 'r') as f1:
            Readargs = f1.readlines()

        # 1. save arguments used
        with open(f'{log_files_folder_path}/args_used.txt', 'w') as f:
            f.write("Command line arguments used -\n")
            f.write(' '.join(sys.argv[1:]))
            f.write("\n\nAll arguments used -\n")
            for data in Readargs:
                f.write(data)

        # 2. create network_snapshot folder
        if save_network_snapshots:
            network_snapshot_save_path = f"{NETWORK_SNAPSHOTS_BASE_FOLDER}/{date_time}"
            os.mkdir(network_snapshot_save_path)

        # 3. assign system variables
        # for demonstration purposes, this reward is for every rewarded action
        rewards = unit_reward

        # 4. get number of roles needed in the network
        roles_requirement = hard_assign.split(',')
        # determine roles to assign
        try:
            workers_needed = int(roles_requirement[0])
        except:
            workers_needed = 1
        try:
            validators_needed = int(roles_requirement[1])
        except:
            validators_needed = 1
        try:
            miners_needed = int(roles_requirement[2])
        except:
            miners_needed = 1

        # 5. check arguments eligibility

        num_devices = num_enterprise
        num_malicious = num_malicious

        if num_devices < workers_needed + miners_needed + validators_needed:
            sys.exit(
                "ERROR: Roles assigned to the devices exceed the maximum number of allowed devices in the network.")

        if num_devices < 3:
            sys.exit(
                "ERROR: There are not enough devices in the network.\n The system needs at least one miner, one worker and/or one validator to start the operation.\nSystem aborted.")

        if num_malicious:
            if num_malicious > num_devices:
                sys.exit(
                    "ERROR: The number of malicious nodes cannot exceed the total number of devices set in this network")
            else:
                print(
                    f"Malicious nodes vs total devices set to {num_malicious}/{num_devices} = {(num_malicious / num_devices) * 100:.2f}%")

        # FedSaC code
        seed = 0
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        random.seed(seed)
        # pca = PCA(n_components=k_principal)
        dw = []
        cluster_model_vectors = {}

        # 6. create neural net based on the input model name

        if model_name == 'mnist_2nn':
            net = Mnist_2NN()
        elif model_name == 'mnist_cnn':
            net = Mnist_CNN2()

        # net = simplecnn(10)

        # 7. assign GPU(s) if available to the net, otherwise CPU
        # os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
        print(f"{torch.cuda.device_count()} GPUs are available to use!")
        print('Current use {}'.format(dev))
        net = net.to(dev)

        # 8. set loss_function
        loss_func = F.cross_entropy

        # 9. create devices in the network
        devices_in_network = DevicesInNetwork(data_set_name='mnist', is_iid=IID, batch_size=batch_size,
                                              learning_rate=learning_rate, loss_func=loss_func,
                                              opti=optimizer, num_devices=num_devices,
                                              network_stability=network_stability, net=net, dev=dev,
                                              knock_out_rounds=knock_out_rounds,
                                              lazy_worker_knock_out_rounds=lazy_local_enterprise_knock_out_rounds,
                                              shard_test_data=shard_test_data,
                                              miner_acception_wait_time=miner_acception_wait_time,
                                              miner_accepted_transactions_size_limit=miner_accepted_transactions_size_limit,
                                              validator_threshold=validator_threshold,
                                              pow_difficulty=pow_difficulty,
                                              even_link_speed_strength=even_link_speed_strength,
                                              base_data_transmission_speed=base_data_transmission_speed,
                                              even_computation_power=even_computation_power,
                                              malicious_updates_discount=malicious_updates_discount,
                                              num_malicious=num_malicious, noise_variance=noise_variance,
                                              check_signature=check_signature,
                                              not_resync_chain=destroy_tx_in_block)
        del net
        devices_list = list(devices_in_network.devices_set.values())

        # 10. register devices and initialize global parameterms
        for device in devices_list:
            # set initial global weights
            device.init_global_parameters()
            # init global model
            device.init_global_model()
            # helper function for registration simulation - set devices_list and aio
            device.set_devices_dict_and_aio(devices_in_network.devices_set, all_in_one)
            # simulate peer registration, with respect to device idx order
            device.register_in_the_network()

        # remove its own from peer list if there is
        for device in devices_list:
            device.remove_peers(device)

        # 11. build logging files/database path
        # create log files
        open(f"{log_files_folder_path}/correctly_kicked_workers.txt", 'w').close()
        open(f"{log_files_folder_path}/mistakenly_kicked_workers.txt", 'w').close()
        open(f"{log_files_folder_path}/false_positive_malious_nodes_inside_slipped.txt", 'w').close()
        open(f"{log_files_folder_path}/false_negative_good_nodes_inside_victims.txt", 'w').close()
        # open(f"{log_files_folder_path}/correctly_kicked_validators.txt", 'w').close()
        # open(f"{log_files_folder_path}/mistakenly_kicked_validators.txt", 'w').close()
        open(f"{log_files_folder_path}/kicked_lazy_workers.txt", 'w').close()

        # 12. setup the mining consensus
        mining_consensus = 'PoW' if pow_difficulty else 'PoS'

    # create malicious worker identification database
    conn = sqlite3.connect(f'{log_files_folder_path}/malicious_wokrer_identifying_log.db')
    conn_cursor = conn.cursor()
    conn_cursor.execute("""CREATE TABLE if not exists  malicious_workers_log (
	device_seq text,
	if_malicious integer,
	correctly_identified_by text,
	incorrectly_identified_by text,
	in_round integer,
	when_resyncing text
	)""")

    target_accuracy = target_acc

    '''
    定义总通信成本、总计算成本、总精度
    '''
    communication_bytes_sum = 0
    computation_sum = 0
    total_accuracy = 0

    # FedLZA starts here
    for comm_round in range(latest_round_num + 1, max_num_comm + 1):
        communication_bytes_per_round = 0
        # create round specific log folder
        log_files_folder_path_comm_round = f"{log_files_folder_path}/comm_{comm_round}"
        if os.path.exists(log_files_folder_path_comm_round):
            print(f"Deleting {log_files_folder_path_comm_round} and create a new one.")
            shutil.rmtree(log_files_folder_path_comm_round)
        os.mkdir(log_files_folder_path_comm_round)
        # free cuda memory
        if dev == torch.device("cuda"):
            with torch.cuda.device('cuda'):
                torch.cuda.empty_cache()
        print(
            f"\n*************************************Communication round {comm_round}****************************************")
        comm_round_start_time = time.time()
        # (RE)ASSIGN ROLES
        workers_to_assign = workers_needed
        miners_to_assign = miners_needed
        validators_to_assign = validators_needed
        workers_this_round = []
        miners_this_round = []
        validators_this_round = []
        random.shuffle(devices_list)
        for device in devices_list:
            if workers_to_assign:
                device.assign_worker_role()
                workers_to_assign -= 1
            elif miners_to_assign:
                device.assign_miner_role()
                miners_to_assign -= 1
            elif validators_to_assign:
                device.assign_validator_role()
                validators_to_assign -= 1
            else:
                device.assign_role()
            if device.return_role() == 'worker':
                workers_this_round.append(device)
            elif device.return_role() == 'miner':
                miners_this_round.append(device)
            else:
                validators_this_round.append(device)
            # determine if online at the beginning (essential for step 1 when worker needs to associate with an online device)
            device.online_switcher()

        ''' DEBUGGING CODE '''
        if verbose:

            # show devices initial chain length and if online
            for device in devices_list:
                if device.is_online():
                    print(f'{device.return_idx()} {device.return_role()} online - ', end='')
                else:
                    print(f'{device.return_idx()} {device.return_role()} offline - ', end='')
                # debug chain length
                print(f"chain length {device.return_blockchain_object().return_chain_length()}")

            # show device roles
            print(
                f"\nThere are {len(workers_this_round)} workers, {len(miners_this_round)} miners and {len(validators_this_round)} validators in this round.")
            print("\nworkers this round are")
            for worker in workers_this_round:
                print(
                    f"d_{worker.return_idx().split('_')[-1]} online - {worker.is_online()} with chain len {worker.return_blockchain_object().return_chain_length()}")
            print("\nminers this round are")
            for miner in miners_this_round:
                print(
                    f"d_{miner.return_idx().split('_')[-1]} online - {miner.is_online()} with chain len {miner.return_blockchain_object().return_chain_length()}")
            print("\nvalidators this round are")
            for validator in validators_this_round:
                print(
                    f"d_{validator.return_idx().split('_')[-1]} online - {validator.is_online()} with chain len {validator.return_blockchain_object().return_chain_length()}")
            print()

            # show peers with round number
            print(f"+++++++++ Round {comm_round} Beginning Peer Lists +++++++++")
            for device_seq, device in devices_in_network.devices_set.items():
                peers = device.return_peers()
                print(f"d_{device_seq.split('_')[-1]} - {device.return_role()[0]} has peer list ", end='')
                for peer in peers:
                    print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()
            print(f"+++++++++ Round {comm_round} Beginning Peer Lists +++++++++")

        ''' DEBUGGING CODE ENDS '''

        # re-init round vars - in real distributed system, they could still fall behind in comm round, but here we assume they will all go into the next round together, thought device may go offline somewhere in the previous round and their variables were not therefore reset
        for miner in miners_this_round:
            miner.miner_reset_vars_for_new_round()
        for worker in workers_this_round:
            worker.worker_reset_vars_for_new_round()
        for validator in validators_this_round:
            validator.validator_reset_vars_for_new_round()

        # DOESN'T MATTER ANY MORE AFTER TRACKING TIME, but let's keep it - orginal purpose: shuffle the list(for worker, this will affect the order of dataset portions to be trained)
        random.shuffle(workers_this_round)
        random.shuffle(miners_this_round)
        random.shuffle(validators_this_round)

        worker_to_index = {}
        index = 0
        for worker in workers_this_round:
            worker_to_index[worker.return_idx()] = index
            index += 1

        '''FedSaC add local models'''
        local_models = []
        train_dl_length = {}
        for worker_iter in range(len(workers_this_round)):
            worker = workers_this_round[worker_iter]
            worker.set_worker_local_model(worker.get_global_model())
            local_models.append(worker.get_worker_local_model())
            # dw.append({key: torch.zeros_like(value) for key, value in local_models[worker_iter].named_parameters()})
            dw.append({key: torch.zeros_like(value) for key, value in local_models[worker_iter].named_parameters()})
            train_dl_length[worker_iter] = len(worker.train_dl.dataset)

        graph_matrix = torch.ones(len(local_models), len(local_models)) / (len(local_models) - 1)
        graph_matrix[range(len(local_models)), range(len(local_models))] = 0  # 对角线元素为0

        nets_this_round = {worker_to_index[worker.return_idx()]: worker.get_worker_local_model() for worker in
                           workers_this_round}
        # nets_this_round = {worker.return_idx(): local_models[worker.return_idx()] for worker in workers_this_round}
        nets_param_start = {worker_to_index[worker.return_idx()]: copy.deepcopy(worker.get_worker_local_model()) for
                            worker in workers_this_round}
        # nets_param_start = {worker.return_idx(): copy.deepcopy(local_models[worker.return_idx()]) for worker in
        #                     workers_this_round}

        total_data_points = sum(train_dl_length[worker_iter] for worker_iter in range(len(workers_this_round)))
        fed_avg_freqs = {worker_iter: train_dl_length[worker_iter] / total_data_points for worker_iter in
                         range(len(workers_this_round))}

        ''' workers, validators and miners take turns to perform jobs '''

        print(
            ''' Step 1 - workers assign associated miner and validator (and do local updates, but it is implemented in code block of step 2) \n''')
        for worker_iter in range(len(workers_this_round)):
            worker = workers_this_round[worker_iter]
            # resync chain(block could be dropped due to fork from last round)
            if worker.resync_chain(mining_consensus):
                worker.update_model_after_chain_resync(log_files_folder_path_comm_round, conn, conn_cursor)
            communication_bytes_per_round += sys.getsizeof(worker.global_parameters)
            # worker (should) perform local update and associate
            print(
                f"{worker.return_idx()} - worker {worker_iter + 1}/{len(workers_this_round)} will associate with a validator and a miner, if online...")
            # worker associates with a miner to accept finally mined block
            if worker.online_switcher():
                associated_miner = worker.associate_with_device("miner")
                if associated_miner:
                    associated_miner.add_device_to_association(worker)
                else:
                    print(f"Cannot find a qualified miner in {worker.return_idx()} peer list.")
            # worker associates with a validator to send worker transactions
            if worker.online_switcher():
                associated_validator = worker.associate_with_device("validator")
                if associated_validator:
                    associated_validator.add_device_to_association(worker)
                else:
                    print(f"Cannot find a qualified validator in {worker.return_idx()} peer list.")

        print(
            ''' Step 2 - validators accept local updates and broadcast to other validators in their respective peer lists (workers local_updates() are called in this step.\n''')
        principal_list = []
        for validator_iter in range(len(validators_this_round)):
            validator = validators_this_round[validator_iter]
            # resync chain
            if validator.resync_chain(mining_consensus):
                validator.update_model_after_chain_resync(log_files_folder_path, conn, conn_cursor)
            communication_bytes_per_round += sys.getsizeof(validator.global_parameters)
            # associate with a miner to send post validation transactions
            if validator.online_switcher():
                associated_miner = validator.associate_with_device("miner")
                if associated_miner:
                    associated_miner.add_device_to_association(validator)
                else:
                    print(f"Cannot find a qualified miner in validator {validator.return_idx()} peer list.")
            # validator accepts local updates from its workers association
            associated_workers = list(validator.return_associated_workers())
            if not associated_workers:
                print(
                    f"No workers are associated with validator {validator.return_idx()} {validator_iter + 1}/{len(validators_this_round)} for this communication round.")
                continue
            validator_link_speed = validator.return_link_speed()
            print(
                f"{validator.return_idx()} - validator {validator_iter + 1}/{len(validators_this_round)} is accepting workers' updates with link speed {validator_link_speed} bytes/s, if online...")
            # records_dict used to record transmission delay for each epoch to determine the next epoch updates arrival time
            records_dict = dict.fromkeys(associated_workers, None)
            for worker, _ in records_dict.items():
                records_dict[worker] = {}
            # used for arrival time easy sorting for later validator broadcasting (and miners' acception order)
            transaction_arrival_queue = {}
            # workers local_updates() called here as their updates transmission may be restrained by miners' acception time and/or size
            if miner_acception_wait_time:
                print(
                    f"miner wati time is specified as {miner_acception_wait_time} seconds. let each worker do local_updates till time limit")
                for worker_iter in range(len(associated_workers)):
                    local_pca = PCA(n_components=k_principal)
                    worker = associated_workers[worker_iter]
                    if not worker.return_idx() in validator.return_black_list():
                        # TODO here, also add print() for below miner's validators
                        print(
                            f'worker {worker_iter + 1}/{len(associated_workers)} of validator {validator.return_idx()} is doing local updates')
                        total_time_tracker = 0
                        update_iter = 1
                        worker_link_speed = worker.return_link_speed()
                        lower_link_speed = validator_link_speed if validator_link_speed < worker_link_speed else worker_link_speed
                        while total_time_tracker < validator.return_miner_acception_wait_time():
                            # simulate the situation that worker may go offline during model updates transmission to the validator, based on per transaction
                            if worker.online_switcher():
                                local_update_spent_time, principal_list = worker.worker_local_update(rewards,
                                                                                                     log_files_folder_path_comm_round,
                                                                                                     comm_round,
                                                                                                     local_pca,
                                                                                                     cluster_model_vectors)
                                unverified_transaction = worker.return_local_updates_and_signature(comm_round)
                                # size in bytes, usually around 35000 bytes per transaction
                                unverified_transactions_size = getsizeof(str(unverified_transaction))
                                transmission_delay = unverified_transactions_size / lower_link_speed
                                if local_update_spent_time + transmission_delay > validator.return_miner_acception_wait_time():
                                    # last transaction sent passes the acception time window
                                    break
                                records_dict[worker][update_iter] = {}
                                records_dict[worker][update_iter]['local_update_time'] = local_update_spent_time
                                records_dict[worker][update_iter]['transmission_delay'] = transmission_delay
                                records_dict[worker][update_iter][
                                    'local_update_unverified_transaction'] = unverified_transaction
                                records_dict[worker][update_iter][
                                    'local_update_unverified_transaction_size'] = unverified_transactions_size
                                if update_iter == 1:
                                    total_time_tracker = local_update_spent_time + transmission_delay
                                else:
                                    total_time_tracker = total_time_tracker - records_dict[worker][update_iter - 1][
                                        'transmission_delay'] + local_update_spent_time + transmission_delay
                                records_dict[worker][update_iter]['arrival_time'] = total_time_tracker
                                if validator.online_switcher():
                                    # accept this transaction only if the validator is online
                                    print(f"validator {validator.return_idx()} has accepted this transaction.")
                                    transaction_arrival_queue[total_time_tracker] = unverified_transaction
                                else:
                                    print(
                                        f"validator {validator.return_idx()} offline and unable to accept this transaction")
                            else:
                                # worker goes offline and skip updating for one transaction, wasted the time of one update and transmission
                                wasted_update_time, wasted_update_params = worker.waste_one_epoch_local_update_time(
                                    optimizer)
                                wasted_update_params_size = getsizeof(str(wasted_update_params))
                                wasted_transmission_delay = wasted_update_params_size / lower_link_speed
                                if wasted_update_time + wasted_transmission_delay > validator.return_miner_acception_wait_time():
                                    # wasted transaction "arrival" passes the acception time window
                                    break
                                records_dict[worker][update_iter] = {}
                                records_dict[worker][update_iter]['transmission_delay'] = transmission_delay
                                if update_iter == 1:
                                    total_time_tracker = wasted_update_time + wasted_transmission_delay
                                    print(
                                        f"worker goes offline and wasted {total_time_tracker} seconds for a transaction")
                                else:
                                    total_time_tracker = total_time_tracker - records_dict[worker][update_iter - 1][
                                        'transmission_delay'] + wasted_update_time + wasted_transmission_delay
                            update_iter += 1
            else:
                # did not specify wait time. every associated worker perform specified number of local epochs
                for worker_iter in range(len(associated_workers)):
                    local_pca = PCA(n_components=k_principal)
                    worker = associated_workers[worker_iter]
                    if not worker.return_idx() in validator.return_black_list():
                        print(
                            f'worker {worker_iter + 1}/{len(associated_workers)} of validator {validator.return_idx()} is doing local updates')
                        if worker.online_switcher():
                            local_update_spent_time, orthogonal_basis = worker.worker_local_update(rewards=rewards,
                                                                                                   log_files_folder_path_comm_round=log_files_folder_path_comm_round,
                                                                                                   comm_round=comm_round,
                                                                                                   pca=local_pca,
                                                                                                   cluster_models=cluster_model_vectors,
                                                                                                   local_epochs=default_local_epochs,
                                                                                                   worker_to_index=worker_to_index)
                            principal_list.append(orthogonal_basis)
                            worker_link_speed = worker.return_link_speed()
                            lower_link_speed = validator_link_speed if validator_link_speed < worker_link_speed else worker_link_speed
                            unverified_transaction = worker.return_local_updates_and_signature(comm_round)
                            unverified_transactions_size = getsizeof(str(unverified_transaction))
                            transmission_delay = unverified_transactions_size / lower_link_speed
                            if validator.online_switcher():
                                transaction_arrival_queue[
                                    local_update_spent_time + transmission_delay] = unverified_transaction
                                print(f"validator {validator.return_idx()} has accepted this transaction.")
                            else:
                                print(
                                    f"validator {validator.return_idx()} offline and unable to accept this transaction")
                        else:
                            print(f"worker {worker.return_idx()} offline and unable do local updates")
                    else:
                        print(
                            f"worker {worker.return_idx()} in validator {validator.return_idx()}'s black list. This worker's transactions won't be accpeted.")
            validator.set_unordered_arrival_time_accepted_worker_transactions(transaction_arrival_queue)
            # in case validator off line for accepting broadcasted transactions but can later back online to validate the transactions itself receives
            validator.set_transaction_for_final_validating_queue(sorted(transaction_arrival_queue.items()))

            global_parameters = validator.get_global_model().state_dict()

            # broadcast to other validators
            if transaction_arrival_queue:
                validator.validator_broadcast_worker_transactions()
            else:
                print(
                    "No transactions have been received by this validator, probably due to workers and/or validators offline or timeout while doing local updates or transmitting updates, or all workers are in validator's black list.")

        graph_matrix = update_graph_matrix_neighbor(graph_matrix=graph_matrix, nets_this_round=nets_this_round,
                                                    initial_global_parameters=global_parameters,
                                                    principal_list=principal_list,
                                                    dw=dw, fed_avg_freqs=fed_avg_freqs, lambda_1=0.9, lambda_2=1.4,
                                                    complementary_metric='PA', similarity_metric='all')

        print(
            ''' Step 2.5 - with the broadcasted workers transactions, validators decide the final transaction arrival order \n''')
        for validator_iter in range(len(validators_this_round)):
            validator = validators_this_round[validator_iter]
            accepted_broadcasted_validator_transactions = validator.return_accepted_broadcasted_worker_transactions()
            print(
                f"{validator.return_idx()} - validator {validator_iter + 1}/{len(validators_this_round)} is calculating the final transactions arrival order by combining the direct worker transactions received and received broadcasted transactions...")
            accepted_broadcasted_transactions_arrival_queue = {}
            if accepted_broadcasted_validator_transactions:
                # calculate broadcasted transactions arrival time
                self_validator_link_speed = validator.return_link_speed()
                for broadcasting_validator_record in accepted_broadcasted_validator_transactions:
                    broadcasting_validator_link_speed = broadcasting_validator_record['source_validator_link_speed']
                    lower_link_speed = self_validator_link_speed if self_validator_link_speed < broadcasting_validator_link_speed else broadcasting_validator_link_speed
                    for arrival_time_at_broadcasting_validator, broadcasted_transaction in \
                            broadcasting_validator_record['broadcasted_transactions'].items():
                        transmission_delay = getsizeof(str(broadcasted_transaction)) / lower_link_speed
                        accepted_broadcasted_transactions_arrival_queue[
                            transmission_delay + arrival_time_at_broadcasting_validator] = broadcasted_transaction
            else:
                print(
                    f"validator {validator.return_idx()} {validator_iter + 1}/{len(validators_this_round)} did not receive any broadcasted worker transaction this round.")
            # mix the boardcasted transactions with the direct accepted transactions
            final_transactions_arrival_queue = sorted(
                {**validator.return_unordered_arrival_time_accepted_worker_transactions(),
                 **accepted_broadcasted_transactions_arrival_queue}.items())
            validator.set_transaction_for_final_validating_queue(final_transactions_arrival_queue)
            print(
                f"{validator.return_idx()} - validator {validator_iter + 1}/{len(validators_this_round)} done calculating the ordered final transactions arrival order. Total {len(final_transactions_arrival_queue)} accepted transactions.")

        print(
            ''' Step 3 - validators do self and cross-validation(validate local updates from workers) by the order of transaction arrival time.\n''')
        for validator_iter in range(len(validators_this_round)):
            validator = validators_this_round[validator_iter]
            final_transactions_arrival_queue = validator.return_final_transactions_validating_queue()
            if final_transactions_arrival_queue:
                # validator asynchronously does one epoch of update and validate on its own test set
                local_validation_time = validator.validator_update_model_by_one_epoch_and_validate_local_accuracy(
                    optimizer)
                print(
                    f"{validator.return_idx()} - validator {validator_iter + 1}/{len(validators_this_round)} is validating received worker transactions...")
                for (arrival_time, unconfirmmed_transaction) in final_transactions_arrival_queue:
                    if validator.online_switcher():
                        # validation won't begin until validator locally done one epoch of update and validation(worker transactions will be queued)
                        if arrival_time < local_validation_time:
                            arrival_time = local_validation_time
                        validation_time, post_validation_unconfirmmed_transaction = validator.validate_worker_transaction(
                            unconfirmmed_transaction, rewards, log_files_folder_path, comm_round,
                            malicious_validator_on)
                        if validation_time:
                            validator.add_post_validation_transaction_to_queue((arrival_time + validation_time,
                                                                                validator.return_link_speed(),
                                                                                post_validation_unconfirmmed_transaction))
                            print(
                                f"A validation process has been done for the transaction from worker {post_validation_unconfirmmed_transaction['worker_device_idx']} by validator {validator.return_idx()}")
                    else:
                        print(
                            f"A validation process is skipped for the transaction from worker {post_validation_unconfirmmed_transaction['worker_device_idx']} by validator {validator.return_idx()} due to validator offline.")
            else:
                print(
                    f"{validator.return_idx()} - validator {validator_iter + 1}/{len(validators_this_round)} did not receive any transaction from worker or validator in this round.")

        print(
            ''' Step 4 - validators send post validation transactions to associated miner and miner broadcasts these to other miners in their respecitve peer lists\n''')
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            # resync chain
            if miner.resync_chain(mining_consensus):
                miner.update_model_after_chain_resync(log_files_folder_path, conn, conn_cursor)
            print(
                f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} accepting validators' post-validation transactions...")
            associated_validators = list(miner.return_associated_validators())
            if not associated_validators:
                print(f"No validators are associated with miner {miner.return_idx()} for this communication round.")
                continue
            self_miner_link_speed = miner.return_link_speed()
            validator_transactions_arrival_queue = {}
            for validator_iter in range(len(associated_validators)):
                validator = associated_validators[validator_iter]
                print(
                    f"{validator.return_idx()} - validator {validator_iter + 1}/{len(associated_validators)} of miner {miner.return_idx()} is sending signature verified transaction...")
                post_validation_transactions_by_validator = validator.return_post_validation_transactions_queue()
                post_validation_unconfirmmed_transaction_iter = 1
                for (validator_sending_time, source_validator_link_spped,
                     post_validation_unconfirmmed_transaction) in post_validation_transactions_by_validator:
                    if validator.online_switcher() and miner.online_switcher():
                        lower_link_speed = self_miner_link_speed if self_miner_link_speed < source_validator_link_spped else source_validator_link_spped
                        transmission_delay = getsizeof(str(post_validation_unconfirmmed_transaction)) / lower_link_speed
                        validator_transactions_arrival_queue[
                            validator_sending_time + transmission_delay] = post_validation_unconfirmmed_transaction
                        print(
                            f"miner {miner.return_idx()} has accepted {post_validation_unconfirmmed_transaction_iter}/{len(post_validation_transactions_by_validator)} post-validation transaction from validator {validator.return_idx()}")
                    else:
                        print(
                            f"miner {miner.return_idx()} has not accepted {post_validation_unconfirmmed_transaction_iter}/{len(post_validation_transactions_by_validator)} post-validation transaction from validator {validator.return_idx()} due to one of devices or both offline.")
                    post_validation_unconfirmmed_transaction_iter += 1
            miner.set_unordered_arrival_time_accepted_validator_transactions(validator_transactions_arrival_queue)
            miner.miner_broadcast_validator_transactions()

        print(
            ''' Step 4.5 - with the broadcasted validator transactions, miners decide the final transaction arrival order\n ''')
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            accepted_broadcasted_validator_transactions = miner.return_accepted_broadcasted_validator_transactions()
            self_miner_link_speed = miner.return_link_speed()
            print(
                f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} calculating the final transactions arrival order by combining the direct worker transactions received and received broadcasted transactions...")
            accepted_broadcasted_transactions_arrival_queue = {}
            if accepted_broadcasted_validator_transactions:
                # calculate broadcasted transactions arrival time
                for broadcasting_miner_record in accepted_broadcasted_validator_transactions:
                    broadcasting_miner_link_speed = broadcasting_miner_record['source_device_link_speed']
                    lower_link_speed = self_miner_link_speed if self_miner_link_speed < broadcasting_miner_link_speed else broadcasting_miner_link_speed
                    for arrival_time_at_broadcasting_miner, broadcasted_transaction in broadcasting_miner_record[
                        'broadcasted_transactions'].items():
                        transmission_delay = getsizeof(str(broadcasted_transaction)) / lower_link_speed
                        accepted_broadcasted_transactions_arrival_queue[
                            transmission_delay + arrival_time_at_broadcasting_miner] = broadcasted_transaction
            else:
                print(
                    f"miner {miner.return_idx()} {miner_iter + 1}/{len(miners_this_round)} did not receive any broadcasted validator transaction this round.")
            # mix the boardcasted transactions with the direct accepted transactions
            final_transactions_arrival_queue = sorted(
                {**miner.return_unordered_arrival_time_accepted_validator_transactions(),
                 **accepted_broadcasted_transactions_arrival_queue}.items())
            miner.set_candidate_transactions_for_final_mining_queue(final_transactions_arrival_queue)
            print(
                f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} done calculating the ordered final transactions arrival order. Total {len(final_transactions_arrival_queue)} accepted transactions.")

        print(
            ''' Step 5 - miners do self and cross-verification (verify validators' signature) by the order of transaction arrival time and record the transactions in the candidate block according to the limit size. Also mine and propagate the block.\n''')
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            final_transactions_arrival_queue = miner.return_final_candidate_transactions_mining_queue()
            valid_validator_sig_candidate_transacitons = []
            invalid_validator_sig_candidate_transacitons = []
            begin_mining_time = 0
            new_begin_mining_time = begin_mining_time
            if final_transactions_arrival_queue:
                print(
                    f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} is verifying received validator transactions...")
                time_limit = miner.return_miner_acception_wait_time()
                size_limit = miner.return_miner_accepted_transactions_size_limit()
                for (arrival_time, unconfirmmed_transaction) in final_transactions_arrival_queue:
                    print('--------------------- new transaction start ---------------------')
                    print('This transaction from worker {} is verified by validator {}'.format(
                        unconfirmmed_transaction['worker_device_idx'], unconfirmmed_transaction['validation_done_by']))
                    if miner.online_switcher():
                        if time_limit:
                            if arrival_time > time_limit:
                                break
                        if size_limit:
                            if getsizeof(
                                    str(valid_validator_sig_candidate_transacitons + invalid_validator_sig_candidate_transacitons)) > size_limit:
                                break
                        # verify validator signature of this transaction
                        verification_time, is_validator_sig_valid = miner.verify_validator_transaction(
                            unconfirmmed_transaction)
                        if verification_time:
                            if is_validator_sig_valid:
                                validator_info_this_tx = {
                                    'validator': unconfirmmed_transaction['validation_done_by'],
                                    'validation_rewards': unconfirmmed_transaction['validation_rewards'],
                                    'validation_time': unconfirmmed_transaction['validation_time'],
                                    'validator_rsa_pub_key': unconfirmmed_transaction['validator_rsa_pub_key'],
                                    'validator_signature': unconfirmmed_transaction['validator_signature'],
                                    'update_direction': unconfirmmed_transaction['update_direction'],
                                    'miner_device_idx': miner.return_idx(),
                                    'miner_verification_time': verification_time,
                                    'miner_rewards_for_this_tx': rewards}
                                # validator's transaction signature valid
                                found_same_worker_transaction = False
                                for valid_validator_sig_candidate_transaciton in valid_validator_sig_candidate_transacitons:
                                    if valid_validator_sig_candidate_transaciton['worker_signature'] == \
                                            unconfirmmed_transaction['worker_signature']:
                                        found_same_worker_transaction = True
                                        break
                                if not found_same_worker_transaction:
                                    valid_validator_sig_candidate_transaciton = copy.deepcopy(unconfirmmed_transaction)
                                    del valid_validator_sig_candidate_transaciton['validation_done_by']
                                    del valid_validator_sig_candidate_transaciton['validation_rewards']
                                    del valid_validator_sig_candidate_transaciton['update_direction']
                                    del valid_validator_sig_candidate_transaciton['validation_time']
                                    del valid_validator_sig_candidate_transaciton['validator_rsa_pub_key']
                                    del valid_validator_sig_candidate_transaciton['validator_signature']
                                    valid_validator_sig_candidate_transaciton['positive_direction_validators'] = []
                                    valid_validator_sig_candidate_transaciton['negative_direction_validators'] = []
                                    valid_validator_sig_candidate_transacitons.append(
                                        valid_validator_sig_candidate_transaciton)
                                if unconfirmmed_transaction['update_direction']:
                                    valid_validator_sig_candidate_transaciton['positive_direction_validators'].append(
                                        validator_info_this_tx)
                                else:
                                    valid_validator_sig_candidate_transaciton['negative_direction_validators'].append(
                                        validator_info_this_tx)
                                transaction_to_sign = valid_validator_sig_candidate_transaciton
                            else:
                                # validator's transaction signature invalid
                                invalid_validator_sig_candidate_transaciton = copy.deepcopy(unconfirmmed_transaction)
                                invalid_validator_sig_candidate_transaciton[
                                    'miner_verification_time'] = verification_time
                                invalid_validator_sig_candidate_transaciton['miner_rewards_for_this_tx'] = rewards
                                invalid_validator_sig_candidate_transacitons.append(
                                    invalid_validator_sig_candidate_transaciton)
                                transaction_to_sign = invalid_validator_sig_candidate_transaciton
                            # (re)sign this candidate transaction
                            signing_time = miner.sign_candidate_transaction(transaction_to_sign)
                            new_begin_mining_time = arrival_time + verification_time + signing_time
                    else:
                        print(
                            f"A verification process is skipped for the transaction from validator {unconfirmmed_transaction['validation_done_by']} by miner {miner.return_idx()} due to miner offline.")
                        new_begin_mining_time = arrival_time
                    begin_mining_time = new_begin_mining_time if new_begin_mining_time > begin_mining_time else begin_mining_time
                transactions_to_record_in_block = {}
                transactions_to_record_in_block[
                    'valid_validator_sig_transacitons'] = valid_validator_sig_candidate_transacitons
                transactions_to_record_in_block[
                    'invalid_validator_sig_transacitons'] = invalid_validator_sig_candidate_transacitons
                # put transactions into candidate block and begin mining
                # block index starts from 1
                start_time_point = time.time()
                candidate_block = Block(idx=miner.return_blockchain_object().return_chain_length() + 1,
                                        transactions=transactions_to_record_in_block,
                                        miner_rsa_pub_key=miner.return_rsa_pub_key())
                # mine the block
                miner_computation_power = miner.return_computation_power()
                if not miner_computation_power:
                    block_generation_time_spent = float('inf')
                    miner.set_block_generation_time_point(float('inf'))
                    print(f"{miner.return_idx()} - miner mines a block in INFINITE time...")
                    continue
                recorded_transactions = candidate_block.return_transactions()
                if recorded_transactions['valid_validator_sig_transacitons'] or recorded_transactions[
                    'invalid_validator_sig_transacitons']:
                    print(f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} mining the block...")
                    # return the last block and add previous hash
                    last_block = miner.return_blockchain_object().return_last_block()
                    if last_block is None:
                        # will mine the genesis block
                        candidate_block.set_previous_block_hash(None)
                    else:
                        candidate_block.set_previous_block_hash(last_block.compute_hash(hash_entire_block=True))
                    # mine the candidate block by PoW, inside which the block_hash is also set
                    mined_block = miner.mine_block(candidate_block, rewards)
                else:
                    print("No transaction to mine for this block.")
                    continue
                # unfortunately may go offline while propagating its block
                if miner.online_switcher():
                    # sign the block
                    miner.sign_block(mined_block)
                    miner.set_mined_block(mined_block)
                    # record mining time
                    block_generation_time_spent = (time.time() - start_time_point) / miner_computation_power
                    miner.set_block_generation_time_point(begin_mining_time + block_generation_time_spent)
                    print(f"{miner.return_idx()} - miner mines a block in {block_generation_time_spent} seconds.")
                    # immediately propagate the block
                    miner.propagated_the_block(miner.return_block_generation_time_point(), mined_block)
                else:
                    print(
                        f"Unfortunately, {miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} goes offline after, if successful, mining a block. This if-successful-mined block is not propagated.")
            else:
                print(
                    f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} did not receive any transaction from validator or miner in this round.")

        print(
            ''' Step 6 - miners decide if adding a propagated block or its own mined block as the legitimate block, and request its associated devices to download this block''')
        forking_happened = False
        # comm_round_block_gen_time regarded as the time point when the winning miner mines its block, calculated from the beginning of the round. If there is forking in PoW or rewards info out of sync in PoS, this time is the avg time point of all the appended time by any device
        comm_round_block_gen_time = []
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            unordered_propagated_block_processing_queue = miner.return_unordered_propagated_block_processing_queue()
            # add self mined block to the processing queue and sort by time
            this_miner_mined_block = miner.return_mined_block()
            if this_miner_mined_block:
                unordered_propagated_block_processing_queue[
                    miner.return_block_generation_time_point()] = this_miner_mined_block
            ordered_all_blocks_processing_queue = sorted(unordered_propagated_block_processing_queue.items())
            if ordered_all_blocks_processing_queue:
                if mining_consensus == 'PoW':
                    print("\nselect winning block based on PoW")
                    # abort mining if propagated block is received
                    print(
                        f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} is deciding if a valid propagated block arrived before it successfully mines its own block...")
                    for (block_arrival_time, block_to_verify) in ordered_all_blocks_processing_queue:
                        verified_block, verification_time = miner.verify_block(block_to_verify,
                                                                               block_to_verify.return_mined_by())
                        if verified_block:
                            block_mined_by = verified_block.return_mined_by()
                            if block_mined_by == miner.return_idx():
                                print(f"Miner {miner.return_idx()} is adding its own mined block.")
                            else:
                                print(
                                    f"Miner {miner.return_idx()} will add a propagated block mined by miner {verified_block.return_mined_by()}.")
                            if miner.online_switcher():
                                miner.add_block(verified_block)
                            else:
                                print(
                                    f"Unfortunately, miner {miner.return_idx()} goes offline while adding this block to its chain.")
                            if miner.return_the_added_block():
                                # requesting devices in its associations to download this block
                                miner.request_to_download(verified_block, block_arrival_time + verification_time)
                                break
                else:
                    # PoS
                    candidate_PoS_blocks = {}
                    print("select winning block based on PoS")
                    # filter the ordered_all_blocks_processing_queue to contain only the blocks within time limit
                    for (block_arrival_time, block_to_verify) in ordered_all_blocks_processing_queue:
                        if block_arrival_time < miner_pos_propagated_block_wait_time:
                            candidate_PoS_blocks[devices_in_network.devices_set[
                                block_to_verify.return_mined_by()].return_stake()] = block_to_verify
                    high_to_low_stake_ordered_blocks = sorted(candidate_PoS_blocks.items(), reverse=True)
                    # for PoS, requests every device in the network to add a valid block that has the most miner stake in the PoS candidate blocks list, which can be verified through chain
                    for (stake, PoS_candidate_block) in high_to_low_stake_ordered_blocks:
                        verified_block, verification_time = miner.verify_block(PoS_candidate_block,
                                                                               PoS_candidate_block.return_mined_by())
                        if verified_block:
                            block_mined_by = verified_block.return_mined_by()
                            if block_mined_by == miner.return_idx():
                                print(f"Miner {miner.return_idx()} with stake {stake} is adding its own mined block.")
                            else:
                                print(
                                    f"Miner {miner.return_idx()} will add a propagated block mined by miner {verified_block.return_mined_by()} with stake {stake}.")
                            if miner.online_switcher():
                                miner.add_block(verified_block)
                            else:
                                print(
                                    f"Unfortunately, miner {miner.return_idx()} goes offline while adding this block to its chain.")
                            if miner.return_the_added_block():
                                # requesting devices in its associations to download this block
                                miner.request_to_download(verified_block, block_arrival_time + verification_time)
                                break
                miner.add_to_round_end_time(block_arrival_time + verification_time)
            else:
                print(
                    f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} does not receive a propagated block and has not mined its own block yet.")
        # CHECK FOR FORKING
        added_blocks_miner_set = set()
        for device in devices_list:
            the_added_block = device.return_the_added_block()
            if the_added_block:
                print(
                    f"{device.return_role()} {device.return_idx()} has added a block mined by {the_added_block.return_mined_by()}")
                added_blocks_miner_set.add(the_added_block.return_mined_by())
                block_generation_time_point = devices_in_network.devices_set[
                    the_added_block.return_mined_by()].return_block_generation_time_point()
                # commented, as we just want to plot the legitimate block gen time, and the wait time is to avoid forking. Also the logic is wrong. Should track the time to the slowest worker after its global model update
                # if mining_consensus == 'PoS':
                # 	if args['miner_pos_propagated_block_wait_time'] != float("inf"):
                # 		block_generation_time_point += args['miner_pos_propagated_block_wait_time']
                comm_round_block_gen_time.append(block_generation_time_point)
        if len(added_blocks_miner_set) > 1:
            print("WARNING: a forking event just happened!")
            forking_happened = True
            with open(f"{log_files_folder_path}/forking_and_no_valid_block_log.txt", 'a') as file:
                file.write(f"Forking in round {comm_round}\n")
        else:
            print("No forking event happened.")

        print(
            ''' Step 6 last step - process the added block - 1.collect usable updated params\n 2.malicious nodes identification\n 3.get rewards\n 4.do local udpates\n This code block is skipped if no valid block was generated in this round''')
        all_devices_round_ends_time = []
        cluster_model_vector_add = defaultdict(list)
        for device in devices_list:
            print('{} is performing final processing'.format(device.return_idx()))
            if device.return_the_added_block() and device.online_switcher():
                # collect usable updated params, malicious nodes identification, get rewards and do local udpates
                processing_time, cluster_model_vector = device.process_block(device.return_the_added_block(),
                                                                             log_files_folder_path, conn,
                                                                             conn_cursor, graph_matrix,
                                                                             nets_this_round, global_parameters,
                                                                             worker_to_index)
                for key, value in cluster_model_vector.items():
                    cluster_model_vector_add[key].append(value.cpu().numpy())
                device.other_tasks_at_the_end_of_comm_round(comm_round, log_files_folder_path)
                device.add_to_round_end_time(processing_time)
                all_devices_round_ends_time.append(device.return_round_end_time())
            else:
                print('{} should not be selected in this round'.format(device.return_idx()))

        for key, value in cluster_model_vector_add.items():
            stack_arrays = np.stack(value)
            average_array = np.mean(stack_arrays, axis=0)
            average_tensor = torch.from_numpy(average_array)
            cluster_model_vectors[key] = average_tensor

        print(''' Logging Accuracies by Devices ''')
        for device in devices_list:
            device.accuracy_this_round = device.validate_model_weights()
            if total_accuracy < device.accuracy_this_round:
                total_accuracy = device.accuracy_this_round
            with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
                is_malicious_node = "M" if device.return_is_malicious() else "B"
                file.write(
                    f"{device.return_idx()} {device.return_role()} {is_malicious_node}: {device.accuracy_this_round}\n")

        communication_bytes_sum += communication_bytes_per_round

        # logging time, mining_consensus and forking
        # get the slowest device end time
        comm_round_spent_time = time.time() - comm_round_start_time
        with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
            # corner case when all miners in this round are malicious devices so their blocks are rejected
            try:
                comm_round_block_gen_time = max(comm_round_block_gen_time)
                file.write(f"comm_round_block_gen_time: {comm_round_block_gen_time}\n")
                file.write('communication overhead: {} bytes\n'.format(communication_bytes_per_round))
            except:
                no_block_msg = "No valid block has been generated this round."
                print(no_block_msg)
                file.write(f"comm_round_block_gen_time: {no_block_msg}\n")
                with open(f"{log_files_folder_path}/forking_and_no_valid_block_log.txt", 'a') as file2:
                    # TODO this may be caused by "no transaction to mine" for the miner. Forgot to check for block miner's maliciousness in request_to_downlaod()
                    file2.write(f"No valid block in round {comm_round}\n")
            try:
                slowest_round_ends_time = max(all_devices_round_ends_time)
                file.write(f"slowest_device_round_ends_time: {slowest_round_ends_time}\n")
            except:
                # corner case when all transactions are rejected by miners
                file.write("slowest_device_round_ends_time: No valid block has been generated this round.\n")
                with open(f"{log_files_folder_path}/forking_and_no_valid_block_log.txt", 'r+') as file2:
                    no_valid_block_msg = f"No valid block in round {comm_round}\n"
                    if file2.readlines()[-1] != no_valid_block_msg:
                        file2.write(no_valid_block_msg)
            file.write(f"mining_consensus: {mining_consensus} {pow_difficulty}\n")
            file.write(f"forking_happened: {forking_happened}\n")
            file.write(f"comm_round_spent_time_on_this_machine: {comm_round_spent_time}\n")
            computation_sum += comm_round_spent_time
        conn.commit()

        # if no forking, log the block miner
        if not forking_happened:
            legitimate_block = None
            for device in devices_list:
                legitimate_block = device.return_the_added_block()
                if legitimate_block is not None:
                    # skip the device who's been identified malicious and cannot get a block from miners
                    break
            with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
                if legitimate_block is None:
                    file.write("block_mined_by: no valid block generated this round\n")
                else:
                    block_mined_by = legitimate_block.return_mined_by()
                    is_malicious_node = "M" if devices_in_network.devices_set[
                        block_mined_by].return_is_malicious() else "B"
                    file.write(f"block_mined_by: {block_mined_by} {is_malicious_node}\n")
        else:
            with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
                file.write(f"block_mined_by: Forking happened\n")

        with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
            file.write(f"Malicious Enterprises:\n")
            for enterprise in devices_list:
                if enterprise.return_is_malicious():
                    file.write(f"{enterprise.return_idx()}, ")

        print(''' Logging Stake by Devices ''')
        for device in devices_list:
            device.accuracy_this_round = device.validate_model_weights()
            with open(f"{log_files_folder_path_comm_round}/stake_comm_{comm_round}.txt", "a") as file:
                is_malicious_node = "M" if device.return_is_malicious() else "B"
                file.write(
                    f"{device.return_idx()} {device.return_role()} {is_malicious_node}: {device.return_stake()}\n")

        # a temporary workaround to free GPU mem by delete txs stored in the blocks. Not good when need to resync chain
        if destroy_tx_in_block:
            for device in devices_list:
                last_block = device.return_blockchain_object().return_last_block()
                if last_block:
                    last_block.free_tx()

        # save network_snapshot if reaches save frequency
        if save_network_snapshots and (comm_round == 1 or comm_round % save_freq == 0):
            if save_most_recent:
                paths = sorted(Path(network_snapshot_save_path).iterdir(), key=os.path.getmtime)
                if len(paths) > save_most_recent:
                    for _ in range(len(paths) - save_most_recent):
                        # make it 0 byte as os.remove() moves file to the bin but may still take space
                        # https://stackoverflow.com/questions/53028607/how-to-remove-the-file-from-trash-in-drive-in-colab
                        open(paths[_], 'w').close()
                        os.remove(paths[_])
            snapshot_file_path = f"{network_snapshot_save_path}/snapshot_r_{comm_round}"
            print(f"Saving network snapshot to {snapshot_file_path}")
            pickle.dump(devices_in_network, open(snapshot_file_path, "wb"))

        print('-------------------comm_round:{} is done!-----------------------'.format(comm_round))
        print("Total Computation Cost(Seconds):{}".format(computation_sum))
        print("Total Communication Cost(Bytes):{}".format(communication_bytes_sum))
        print("Total Accuracy(%):{}".format(total_accuracy * 100))

        if total_accuracy >= target_accuracy:
            break

    with open(f'{log_files_folder_path}/Output.txt', 'w') as f:
        # FedAnil+: Total Computation Cost (Seconds)
        f.write(f"Total Computation Cost (Seconds): {computation_sum} \n")
        # FedAnil+: Total Communication Cost (Bytes)
        f.write(f"Total Communication Cost (Bytes): {communication_bytes_sum} \n")
        # FedAnil+: Total Accuracy (%)
        f.write(f"Total Accuracy (%): {total_accuracy * 100} \n")
