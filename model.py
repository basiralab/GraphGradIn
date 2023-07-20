import torch
import helper
import random
import uuid
import os
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import NNConv
import time
from torch.nn import Sequential, Linear, ReLU
import pickle
from sklearn.model_selection import KFold
from GraphGradIn import GraphGradIn
from GraphTestIn import GraphTestIn
from config import MODEL_PARAMS, CONFIG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# These two options should be seed to ensure reproducible (If you are using cudnn backend)
# https://pytorch.org/docs/stable/notes/randomness.html
# We used 35813 (part of the Fibonacci Sequence) as the seed when we conducted experiments

seed = 35813

MODEL_WEIGHT_BACKUP_PATH = "./output"
DEEP_CBT_SAVE_PATH = "./output/cbts"
TEMP_FOLDER = "./temp"


class DGN(torch.nn.Module):
    def __init__(self, MODEL_PARAMS):
        super(DGN, self).__init__()
        self.model_params = MODEL_PARAMS

        nn = Sequential(Linear(self.model_params["Linear1"]["in"], self.model_params["Linear1"]["out"]), ReLU())
        self.conv1 = NNConv(self.model_params["conv1"]["in"], self.model_params["conv1"]["out"], nn, aggr='mean')

        nn = Sequential(Linear(self.model_params["Linear2"]["in"], self.model_params["Linear2"]["out"]), ReLU())
        self.conv2 = NNConv(self.model_params["conv2"]["in"], self.model_params["conv2"]["out"], nn, aggr='mean')

        nn = Sequential(Linear(self.model_params["Linear3"]["in"], self.model_params["Linear3"]["out"]), ReLU())
        self.conv3 = NNConv(self.model_params["conv3"]["in"], self.model_params["conv3"]["out"], nn, aggr='mean')

    def forward(self, data):
        """
            Args:
                data (Object): data object consist of three parts x, edge_attr, and edge_index.
                                This object can be produced by using helper.cast_data function
                        x: Node features with shape [number_of_nodes, 1] (Simply set to vector of ones since we dont have any)
                        edge_attr: Edge features with shape [number_of_edges, number_of_views]
                        edge_index: Graph connectivities with shape [2, number_of_edges] (COO format)


        """
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index

        x = F.relu(self.conv1(x, edge_index, edge_attr))

        x = F.relu(self.conv2(x, edge_index, edge_attr))

        x = F.relu(self.conv3(x, edge_index, edge_attr))

        repeated_out = x.repeat(self.model_params["N_ROIs"], 1, 1)
        repeated_t = torch.transpose(repeated_out, 0, 1)
        diff = torch.abs(repeated_out - repeated_t)
        cbt = torch.sum(diff, 2)

        return cbt

    @staticmethod
    def generate_subject_biased_cbts(model, train_data):
        """
            Generates all possible CBTs for a given training set.
            Args:
                model: trained DGN model
                train_data: list of data objects
        """
        model.eval()
        cbts = np.zeros((model.model_params["N_ROIs"], model.model_params["N_ROIs"], len(train_data)))
        train_data = [d.to(device) for d in train_data]
        for i, data in enumerate(train_data):
            cbt = model(data)
            cbts[:, :, i] = np.array(cbt.cpu().detach())
        return cbts

    @staticmethod
    def generate_cbt_median(model, train_data):
        """
            Generate optimized CBT for the training set (use post training refinement)
            Args:
                model: trained DGN model
                train_data: list of data objects
        """
        with torch.no_grad():
            cbts = []
            train_data = [d.to(device) for d in train_data]
            for data in train_data:
                cbt = model(data)
                cbts.append(np.array(cbt.cpu().detach()))
            final_cbt = torch.tensor(np.median(cbts, axis=0), dtype=torch.float32).to(device)

        return final_cbt

    @staticmethod
    def mean_frobenious_distance(generated_cbt, test_data):
        """
            Calculate the mean Frobenious distance between the CBT and test subjects (all views)
            Args:
                generated_cbt: trained DGN model
                test_data: list of data objects
        """
        frobenius_all = []
        for data in test_data:
            views = data.con_mat
            for index in range(views.shape[2]):
                diff = torch.abs(views[:, :, index] - generated_cbt)
                diff = diff * diff
                sum_of_all = diff.sum()
                d = torch.sqrt(sum_of_all)
                frobenius_all.append(d)
        return sum(frobenius_all) / len(frobenius_all)

    @staticmethod
    def train_model(X, model_params, n_max_epochs, early_stop, args, random_sample_size=10,
                    weighted_loss=False, n_folds=5):
        """
            Trains a model for each cross validation fold and
            saves all models along with CBTs to ./output/<model_name>
            Args:
                X (np array): dataset with shape [N_Subjects, N_ROIs, N_ROIs, N_Views]
                model params (dict): model parameters from config.py file
                n_max_epochs (int): number of training epochs (if early_stop == True this is maximum epoch limit)
                early_stop (bool): if set true, model will stop training when overfitting starts.
                args: To select GraphGradIn and GraphTestIn
                random_sample_size (int): random subset size for SNL function
                weighted_loss (bool): view normalization in centeredness loss
                n_folds (int): number of cross validation folds
            Return:
                models: trained models
        """

        k2 = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
        count = 0

        x_casted = [d for d in helper.cast_data(X)]

        influential_losses = []
        standard_losses = []

        for train_indices, test_indices in k2.split(X):

            torch.cuda.empty_cache()
            print("********* FOLD {} *********".format(count+1))

            casted_test = [x_casted[x].to(device) for x in test_indices]
            train_casted = [x_casted[x].to(device) for x in train_indices]

            # CALCULATING THE INFLUENCE SCORES
            if args.GraphGradIn:
                model_name = ["simulated"] + ["GraphGradIn"]
                data_influence_scores = GraphGradIn.calculate_GraphGradIn(
                    X=train_casted,
                    model_params=MODEL_PARAMS,
                    saved_epoch_score_list=CONFIG["saved_epoch_score_list"],
                    model_name=model_name)

            elif args.GraphTestIn:
                model_name = ["simulated"] + ["GraphTestIn"]
                data_influence_scores = GraphTestIn.calculate_GraphTestIn(
                    X=train_casted,
                    model_params=MODEL_PARAMS,
                    saved_epoch_score_list=CONFIG["saved_epoch_score_list"],
                    model_name=model_name)
            else:
                model_name = ["simulated"] + ["GraphGradIn"]
                print(
                    'The influence score calculation method was not entered, GraphGradIn method is selected as default.')
                data_influence_scores = GraphGradIn.calculate_GraphGradIn(
                    X=train_casted,
                    model_params=MODEL_PARAMS,
                    saved_epoch_score_list=CONFIG["saved_epoch_score_list"],
                    model_name=model_name)

            dataset_name = model_name[0]
            method_name = model_name[1]

            model_name = model_name[0] + '_' + model_name[1]
            data_dict = data_influence_scores[count]

            # Sort data dictionary starting from sample with best influence score
            data_dict = {k: v for k, v in sorted(data_dict.items(), reverse=True, key=lambda item: item[1][1])}
            indexes = [idx for _, _, idx in data_dict.values()]

            fold_indices = [x for x in indexes if x in train_indices]

            models = []
            save_path = MODEL_WEIGHT_BACKUP_PATH + "/" + model_name + "/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_path_CBTs = DEEP_CBT_SAVE_PATH + "/" + model_name + "/"
            if not os.path.exists(save_path_CBTs):
                os.makedirs(save_path_CBTs)

            model_id = str(uuid.uuid4())
            with open(save_path + "model_params.txt", 'w') as f:
                print(model_params, file=f)

            train_data, test_data, train_mean, train_std = helper.preprocess_casted_data(train_casted, casted_test)

            if weighted_loss:
                loss_weights = torch.tensor(
                    np.array(list((1 / train_mean) / np.max(1 / train_mean)) * len(train_data)),
                    dtype=torch.float32)
                loss_weights = loss_weights.to(device)

            model = DGN(model_params)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=model_params["learning_rate"], weight_decay=0)
            targets = [torch.tensor(tensor, dtype=torch.float32).to(device) for tensor in train_data]

            test_errors = []
            tick = time.time()
            loss_epoch = []
            CBTs = []
            for epoch in range(n_max_epochs):
                model.train()
                losses = []
                for t, data in enumerate(train_casted):
                    # Compose Dissimilarity matrix from network outputs
                    cbt = model(data)
                    views_sampled = random.sample(targets, random_sample_size)
                    sampled_targets = torch.cat(views_sampled, axis=2).permute((2, 1, 0))
                    expanded_cbt = cbt.expand((sampled_targets.shape[0], 35, 35))

                    # SNL function
                    diff = torch.abs(expanded_cbt - sampled_targets)  # Absolute difference
                    sum_of_all = torch.mul(diff, diff).sum(axis=(1, 2))  # Sum of squares
                    l = torch.sqrt(sum_of_all)  # Square root of the sum
                    if weighted_loss:
                        rep_loss = (l * loss_weights[:random_sample_size * model_params["n_attr"]].mean())
                    else:
                        rep_loss = l.mean()
                    losses.append(rep_loss)

                optimizer.zero_grad()
                loss = torch.mean(torch.stack(losses))
                loss_epoch.append(float(loss))
                loss.backward()
                optimizer.step()

                # Track the loss
                if epoch % 10 == 0:
                    cbt = DGN.generate_cbt_median(model, train_casted)
                    test_loss = DGN.mean_frobenious_distance(cbt, casted_test)
                    rep_loss = float(test_loss)

                    tock = time.time()
                    time_elapsed = tock - tick
                    tick = tock
                    test_errors.append(rep_loss)
                    print("Epoch: {}  |  Test Rep: {:.2f}  | Time Elapsed: {:.2f}  |".format(epoch, rep_loss, time_elapsed))

                    # Early stopping control
                    if len(test_errors) > 6 and early_stop:
                        torch.save(model.state_dict(),
                                   TEMP_FOLDER + "/weight_" + model_id + "_" + str(rep_loss)[:5] + ".model")
                        last_6 = test_errors[-6:]
                        if (all(last_6[i] < last_6[i + 1] for i in range(5))):
                            print("Early Stopping")
                            break

            # Save the model and the CBTS
            torch.save(model.state_dict(), save_path + "fold" + str(count) + "standard" + dataset_name + '_' + model_name + ".model")
            models.append(model)

            cbt_standard = DGN.generate_cbt_median(model, train_casted)
            rep_loss = DGN.mean_frobenious_distance(cbt_standard, casted_test)

            cbt_standard = cbt_standard.cpu().numpy()
            CBTs.append(cbt_standard)
            np.save(save_path_CBTs + "fold" + str(count) + "standard" + "_cbt_" + dataset_name + '_' + model_name, cbt_standard)
            # Save all subject biased CBTs
            all_cbts = DGN.generate_subject_biased_cbts(model, train_casted)
            np.save(save_path_CBTs + "fold" + str(count) + "standard" + "_all_cbts_" + dataset_name + '_' + model_name, all_cbts)
            standard_losses.append(float(rep_loss))

            ###########################################################################################################
            # Using n_p samples with highest scores for CBT generation

            # Taking top %10 to %40 percent of whole dataset (samples with highest influence scores)
            elements_taken = np.linspace(int(len(train_casted) * 0.1), int(len(train_casted) * 0.4), 6).astype(int)

            influential_training_losses = []
            influential_CBTs = []
            for el_taken in elements_taken:
                print(method_name + " for {} top scorers taken".format(el_taken))

                # Constructing influential training dataset using samples with highest influence scores
                influential_training_casted = helper.rebuild_influential_training_dataset(fold_indices, el_taken, train_casted)

                infl_train_data, test_data, train_mean, train_std = helper.preprocess_casted_data(influential_training_casted, casted_test)

                if weighted_loss:
                    loss_weights = torch.tensor(
                        np.array(list((1 / train_mean) / np.max(1 / train_mean)) * len(train_data)),
                        dtype=torch.float32)
                    loss_weights = loss_weights.to(device)

                model_prop = DGN(model_params)
                model_prop = model_prop.to(device)
                optimizer_prop = torch.optim.Adam(model_prop.parameters(), lr=model_params["learning_rate"], weight_decay=0)

                test_errors = []
                tick = time.time()
                loss_epoch_infl_trainingset = []

                for epoch in range(n_max_epochs):
                    model_prop.train()
                    losses_infl_trainingset = []

                    for t, data in enumerate(influential_training_casted):
                        # Compose Dissimilarity matrix from network outputs
                        cbt = model_prop(data)
                        views_sampled = random.sample(targets, random_sample_size)
                        sampled_targets = torch.cat(views_sampled, axis=2).permute((2, 1, 0))
                        expanded_cbt = cbt.expand((sampled_targets.shape[0], 35, 35))

                        # SNL function
                        diff = torch.abs(expanded_cbt - sampled_targets)  # Absolute difference
                        sum_of_all = torch.mul(diff, diff).sum(axis=(1, 2))  # Sum of squares
                        l = torch.sqrt(sum_of_all)  # Square root of the sum
                        if weighted_loss:
                            rep_loss = (l * loss_weights[:random_sample_size * model_params["n_attr"]].mean())
                        else:
                            rep_loss = l.mean()

                        losses_infl_trainingset.append(rep_loss)

                    optimizer_prop.zero_grad()
                    loss_prop = torch.mean(torch.stack(losses_infl_trainingset))
                    loss_epoch_infl_trainingset.append(float(loss_prop))
                    loss_prop.backward()
                    optimizer_prop.step()

                    # Track the loss
                    if epoch % 10 == 0:
                        cbt = DGN.generate_cbt_median(model_prop, influential_training_casted)
                        test_loss = DGN.mean_frobenious_distance(cbt, casted_test)

                        rep_loss = float(test_loss)

                        tock = time.time()
                        time_elapsed = tock - tick
                        tick = tock
                        test_errors.append(rep_loss)
                        print("Epoch: {}  |  Test Rep: {:.2f}  |  Time Elapsed: {:.2f}  |".format(epoch, rep_loss,
                                                                                                  time_elapsed))
                        # Early stopping control
                        if len(test_errors) > 6 and early_stop:
                            torch.save(model_prop.state_dict(),
                                       TEMP_FOLDER + "/weight_prop_" + model_id + "_" + str(rep_loss)[:5] + ".model")
                            last_6 = test_errors[-6:]
                            if (all(last_6[i] < last_6[i + 1] for i in range(5))):
                                print("Early Stopping")
                                break

                # Create improved CBT
                cbt_b = DGN.generate_cbt_median(model_prop, influential_training_casted)
                rep_loss_b = DGN.mean_frobenious_distance(cbt_b, casted_test)

                influential_training_losses.append(float(rep_loss_b))

                cbt_b = cbt_b.cpu().numpy()
                influential_CBTs.append(cbt_b)
                np.save(save_path_CBTs + "fold" + str(count) + "_influential_" + str(el_taken) + "_cbt_" + dataset_name + '_' + model_name,
                        cbt_b)
                # Save all subject biased CBTs
                all_cbts = DGN.generate_subject_biased_cbts(model_prop, influential_training_casted)
                np.save(save_path_CBTs + "fold" + str(count) + "influential" + str(el_taken) + "_all_cbts_" + dataset_name + '_' + model_name,
                        all_cbts)

            influential_losses.append(influential_training_losses)

            print("Fold {} - Using influential training dataset - Centeredness for n_p = {} values: {}".format(count+1, elements_taken, influential_losses[count]))
            print("Fold {} - Averaged centeredness = {}, ".format(count+1, np.mean(influential_losses[count])))
            print("Fold {} - Centeredness for best n_p = {} value: {}, ".format(count+1, elements_taken[np.argmax(influential_losses[count])], np.min(influential_losses[count])))
            print("Fold {} - Using whole training dataset - Centeredness: {}".format(count+1, standard_losses[count]))

            count += 1

            with open('./standard_' + dataset_name + '_' + method_name + '.pickle', 'wb') as f:
                pickle.dump(standard_losses, f)
            with open('./influential_' + dataset_name + '_' + method_name + '.pickle', 'wb') as f:
                pickle.dump(influential_losses, f)

        print("FINAL RESULTS")
        print("Averaged centeredness = {}, ".format(np.mean(influential_losses, axis=1)))
        print("Centeredness for best values: {}, ".format(np.min(influential_losses, axis=1)))
        print("Using whole training dataset - Centeredness: {}".format(standard_losses))
        helper.clear_dir(TEMP_FOLDER)
