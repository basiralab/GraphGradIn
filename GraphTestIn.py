import torch
import helper
import random
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import NNConv
import time
from torch.nn import Sequential, Linear, ReLU
from sklearn.model_selection import KFold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# These two options should be seed to ensure reproducible (If you are using cudnn backend)
# https://pytorch.org/docs/stable/notes/randomness.html
# We used 35813 (part of the Fibonacci Sequence) as the seed when we conducted experiments
seed = 35813
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
TEMP_FOLDER = 'temp'
WEIGHTS_FOLDER = 'saved_weights'


class GraphTestIn(torch.nn.Module):
    def __init__(self, MODEL_PARAMS):
        super(GraphTestIn, self).__init__()
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
                        x: Node features with shape [number_of_nodes, 1] (Simply set to vector of ones since we don't have any)
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
    def generate_cbt_median_with_grad(model, train_data):
        """
            Generate optimized CBT for the training set (use post training refinement) with gradients
            Args:
                model: trained DGN model
                train_data: list of data objects
        """

        cbts = []
        train_data = [d.to(device) for d in train_data]
        for data in train_data:
            cbt = model(data)
            cbts.append(cbt)
        final_cbt = torch.median(torch.stack(cbts), 0)[0].to(device)

        return final_cbt

    @staticmethod
    def mean_frobenious_distance(generated_cbt, val_data):
        """
            Calculate the mean Frobenious distance between the CBT and test subjects (all views)
            Args:
                generated_cbt: trained DGN model
                val_data: list of data objects
        """
        frobenius_all = []
        for data in val_data:
            views = data.con_mat
            for index in range(views.shape[2]):
                diff = torch.abs(views[:, :, index] - generated_cbt)
                diff = diff * diff
                sum_of_all = diff.sum()
                d = torch.sqrt(sum_of_all)
                frobenius_all.append(d)
        return sum(frobenius_all) / len(frobenius_all)

    @staticmethod
    def mean_frobenious_distance_grad(generated_cbt, val_data):
        """
            Calculate the mean Frobenious distance between the CBT and test subjects (all views) with gradients
            Args:
                generated_cbt: trained DGN model
                val_data: list of data objects
        """
        frobenius_all = []
        for data in val_data:
            views = data.con_mat
            for index in range(views.shape[2]):
                diff = torch.abs(views[:, :, index] - generated_cbt)
                diff = diff * diff
                sum_of_all = diff.sum()
                d = torch.sqrt(sum_of_all)
                frobenius_all.append(d)
        return sum(frobenius_all) / len(frobenius_all)

    @staticmethod
    def calculate_GraphTestIn(X, model_params, saved_epoch_score_list, model_name, n_folds=5, random_sample_size=10, weighted_loss=False):
        """
            Calculates GraphTestIn for each cross validation fold and
            Args:
                X (np array): dataset (train+test) with shape [N_Subjects, N_ROIs, N_ROIs, N_Views]
                saved_epoch_score_list (list): list of scores calculated at each epoch, in our case this is the last epoch
                n_folds (int): number of folds for cross-validation
                model_name (list): name for saving the model
                random_sample_size (int): random subset size for SNL function
                weighted_loss (bool): view normalization in centeredness loss
            Return:
                GraphTestIn_scores_list (list): list of dictionary, GraphGradIn scores for training data
        """

        dataset_name = model_name[0]
        method_name = model_name[1]

        count = 0

        k = KFold(n_splits=n_folds, random_state=seed, shuffle=True)

        GraphTestIn_scores_list = []
        time_list = []
        for train_indices, val_indices in k.split(X):

            train_casted = [X[x].to(device) for x in train_indices]
            val_casted = [X[x].to(device) for x in val_indices]

            torch.cuda.empty_cache()
            print("********* FOLD {} *********".format(count+1))

            train_data, val_data, train_mean, train_std = helper.preprocess_casted_data(train_casted, val_casted)

            if weighted_loss:
                loss_weights = torch.tensor(
                    np.array(list((1 / train_mean) / np.max(1 / train_mean)) * len(train_data)), dtype=torch.float32)
                loss_weights = loss_weights.to(device)

            model = GraphTestIn(model_params)
            model = model.to(device)

            # Taking the gradients from last layer
            model.conv3.requires_grad = True

            optimizer = torch.optim.Adam(model.parameters(), lr=model_params["learning_rate"], weight_decay=0)
            targets = [torch.tensor(tensor, dtype=torch.float32).to(device) for tensor in train_data]
            targets_val = [torch.tensor(tensor, dtype=torch.float32).to(device) for tensor in val_data]

            test_errors = []
            for epoch in range(max(saved_epoch_score_list)):
                model.train()

                losses = []
                data_list = []
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
                    data_list.append(data)

                optimizer.zero_grad()
                loss = torch.mean(torch.stack(losses))
                loss.backward()
                optimizer.step()

                # Track the loss
                if epoch % 10 == 0:
                    cbt_b = GraphTestIn.generate_cbt_median(model, train_casted)
                    rep_loss = GraphTestIn.mean_frobenious_distance(cbt_b, val_casted)
                    test_errors.append(rep_loss)
                    print("Epoch: {}  |  Test Rep: {:.2f}  |".format(epoch, rep_loss))

                if (epoch + 1) in saved_epoch_score_list:
                    torch.save({
                        'epoch': (epoch + 1),
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, TEMP_FOLDER + '/model_epoch_' + str((epoch + 1)) + '.pt')
                    print("{}th weights saved!".format((epoch + 1)))

            # Influence score calculation starts
            data_dict = {}
            torch.cuda.empty_cache()
            print("INFLUENCE SCORE CALCULATION FOR FOLD {}".format(count + 1))

            tick = time.time()
            for ep in saved_epoch_score_list:
                new_model = GraphTestIn(model_params)
                new_model = new_model.to(device)
                new_model.conv3.requires_grad = True

                new_optimizer = torch.optim.Adam(new_model.parameters(), lr=model_params["learning_rate"], weight_decay=0)

                epoch_weights = torch.load(TEMP_FOLDER + '/model_epoch_' + str(ep) + '.pt')
                print("{}th weights loaded.".format((epoch + 1)))

                new_model.load_state_dict(epoch_weights['model_state_dict'])
                new_optimizer.load_state_dict(epoch_weights['optimizer_state_dict'])

                new_model.train()

                # Using whole training set in the refinement
                cbt_b = GraphTestIn.generate_cbt_median(new_model, train_casted)
                rep_loss = GraphTestIn.mean_frobenious_distance(cbt_b, val_casted)

                scored_data = []
                for data in train_casted:
                    scored_data.append(data)
                    # Exclude a training point Z and then refine
                    train_casted_exclude = [train_data for train_data in train_casted if train_data != data]
                    cbt_excluded = GraphTestIn.generate_cbt_median(new_model, train_casted_exclude)
                    rep_loss_excluded = GraphTestIn.mean_frobenious_distance(cbt_excluded, val_casted)

                    # GraphTestIn score for training multigraph Z
                    score_excl = rep_loss_excluded - rep_loss

                    # Update score dictionary
                    helper.update_score_dictionary(data_dict, data, score_excl)

            tock = time.time()
            time_elapsed = tock - tick
            print('Time elapsed for GraphTestIn calculation in Fold {}: {}'.format(count + 1, time_elapsed))
            count += 1
            GraphTestIn_scores_list.append(data_dict)
            time_list.append(time_elapsed)

        mean_time = np.mean(time_list)
        std_time = np.std(time_list)

        print('Average fold time: {}, with std {}'.format(mean_time, std_time))

        helper.clear_dir(TEMP_FOLDER)
        #torch.save(GraphTestIn_scores_list, './' + dataset_name + '_' + method_name)
        return GraphTestIn_scores_list
