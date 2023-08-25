from graphviz import Digraph
import numpy as np
from matplotlib import pyplot as plt
import copy
import pandas as pd

MIN_LEAF_SIZE = None
COUNT_NODES = 0
TRAIN_VAL = 0.8
TOTAL = 0
MAX_HEIGHT = None
PATH_ = None

class NODE:
    '''
        Class defines the nodes of the decision tree
    '''

    def __init__(self, attribute, value, Arr_typ):
        '''
            Initializes the node
        '''
        self.indx_attr = attribute
        self.val = value
        self.attr_type = Arr_typ[self.indx_attr]
        self.left = None
        self.right = None
        self.leaf = False
        self.response = None
        global COUNT_NODES
        COUNT_NODES = COUNT_NODES+1
        self.node_id=COUNT_NODES

    def create_leaf(self, response):
        '''
            Makes the node a leaf node
        '''
        # print(f'Making leaf node {response}')
        self.leaf = True
        self.left = None
        self.right = None
        self.response = response

    def select_node(self, X):
        '''
            Predicts the class of the instance X
        '''
        if self.leaf:
            return self.response
        else:
            if self.attr_type == 'cont':
                if X[self.indx_attr] <= self.val:
             
                    return self.left.select_node(X)
                else:
                    return self.right.select_node(X)
            else:
                if X[self.indx_attr] == self.val:
                   
                    return self.left.select_node(X)
                else:
                    return self.right.select_node(X)

    def __eq__(self, other) -> bool:
        if other is None:
            return False
        if self.leaf and other.leaf:
            return self.response == other.response
        if self.indx_attr != other.indx_attr or self.val != other.val or self.attr_type != other.attr_type:
            return False
        if self.left != other.left or self.right != other.right:
            return False
        return True

    def DFS(self):
        count = 1
        if not self.left is None:
            count += self.left.DFS()
        if not self.right is None:
            count += self.right.DFS()
        ans = count
        return ans

    def prune_cal_error(self, X_val, y_val):
        preds = np.array([self.select_node(x) for x in X_val])
        return np.sum(preds != y_val)

    def base_prune(self, train_y, X_val, y_val, n_node):

        leaf = classify(train_y)
        errors_leaf = np.sum(y_val != leaf)
        errors_node = np.sum(y_val != np.array(
            [self.select_node(x) for x in X_val]))

        if errors_leaf <= errors_node:
            n_node.create_leaf(leaf)
        
    def rec_prune(self, train_X, train_y, X_val, y_val,Arr_typ):

        n_node = NODE(self.indx_attr, self.val, Arr_typ)
        n_node.leaf = self.leaf
        n_node.response = self.response
        n_node.left = self.left
        n_node.right = self.right
        if self.leaf:
            n_node = self.base_prune(train_y, X_val, y_val, n_node)

        else:
            X_train_yes, Y_train_yes, X_train_no, Y_train_no = filter(
                train_X, train_y, self.indx_attr, self.val,Arr_typ)
            X_val_yes, Y_val_yes, X_val_no, Y_val_no = filter(
                X_val, y_val, self.indx_attr, self.val,Arr_typ)

            if not (self.left is None or self.left.leaf==True):
                n_node.left = self.left.rec_prune(X_train_yes,Y_train_yes, X_val_yes, Y_val_yes,Arr_typ)
            if not (self.right is None or self.right.leaf==True):
                n_node.right = self.right.rec_prune(X_train_no,Y_train_no, X_val_no, Y_val_no,Arr_typ)

            self.base_prune(train_y, X_val, y_val, n_node)

        return n_node


class DecisionTree:
    '''
        self.root, self.X, self.y
    '''

    def __init__(self, X, y, column_names, min_leaf_size, max_depth):
        '''
            Initializes the tree
        '''
        self.root = None
        self.root_pruned = None
        self.X = X
        self.y = y
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.column_names = column_names
        self.Arr_typ = feature_type(X, 2)

    def fit(self):
        '''
                Builds the decision tree
        '''
        global COUNT_NODES
        COUNT_NODES=0
        self.root = self.build_tree(self.X, self.y)

    def is_leaf(self, X_lo, y_lo):
        '''
            Checks if the node is a leaf
        '''
        if X_lo.shape[0] <= self.min_leaf_size:
            return True
        if purity_check(y_lo):
            return True
        return False

    def build_tree(self, X, y, depth=0):
        '''
            Recursively builds the decision tree
        '''

        
        if self.is_leaf(X, y) or depth == self.max_depth:
            node = NODE(0, 0, self.Arr_typ)
            node.create_leaf(classify(y))
            return node
        else:
            depth += 1
            best_attr, best_val = get_best_split(
                X, y, self.Arr_typ)
            node = NODE(best_attr, best_val, self.Arr_typ)
            
            X_left, y_left, X_right, y_right = children_create(
                X, y, best_attr, best_val, self.Arr_typ)
            
            if X_left.shape[0] == 0 or X_right.shape[0] == 0:
                node.create_leaf(classify(y))    
                return node

            left_tree = self.build_tree(X_left, y_left, depth)
            right_tree = self.build_tree(X_right, y_right, depth)
            
            
            if left_tree == right_tree:    
                node.create_leaf(classify(y_left))
            else:
                node.leaf = False
                node.left = left_tree
                node.right = right_tree

            return node

    def predict(self, X):
        '''
            Predicts the labels of the test data
        '''
        if self.root is None:
            return None
        else:
            return np.array([self.root.select_node(x) for x in X])

    def CALC_ACCURACY(self, X, y):
        '''
            Calculates the accuracy of the decision tree
        '''
        y_pred = self.predict(X)
        from sklearn.metrics import classification_report
        print(classification_report(y, y_pred))
        return calc_accuracy(y, y_pred)

    def calc_pruned_accuracy(self, X, y):
        y_pred = self.predict_prune(X)
        from sklearn.metrics import classification_report
        
        print(classification_report(y, y_pred))
        return calc_accuracy(y, y_pred)

    def COUNT_NODES(self):
        return self.root.DFS()

    def prune_post(self, train_X, train_y, X_val, y_val):
        '''
            Recursively prunes the tree
        '''
        global COUNT_NODES
        COUNT_NODES=0
        self.root_pruned = self.root.rec_prune(train_X, train_y, X_val, y_val,self.Arr_typ)

    def predict_prune(self, X):
        if self.root_pruned is None:
            return None
        else:
            return np.array([self.root_pruned.select_node(x) for x in X])


def render_node(vertex, column_first_names, count):
    if vertex.leaf:
        return f'ID {vertex.node_id},\nresponse = {vertex.response}\n'
    if column_first_names[vertex.indx_attr] == 'Vehicle_Age':
        return f'ID {vertex.node_id}\n{column_first_names[vertex.indx_attr]} -  {vertex.val}\n'
    return f'ID {vertex.node_id}\n{column_first_names[vertex.indx_attr]} <= {vertex.val}\n'


def tree_to_gv(node_root, column_first_names,file_name="decision_tree.gv"):
    f = Digraph('Decision Tree', filename=file_name)

    f.attr('node', shape='rectangle')
    queue = [node_root]
    idx = 0
    index = 0
    while len(queue) > 0:
        index= index + 1
        node = queue.pop(0)
        if node is None:
            continue
        if not node.left is None:
            f.edge(render_node(node, column_first_names, idx), render_node(
                node.left, column_first_names, idx), label='True')
            idx += 1
            queue.append(node.left)
        if not node.right is None:
            f.edge(render_node(node, column_first_names, idx), render_node(
                node.right, column_first_names, idx), label='False')
            idx += 1
            queue.append(node.right)
    f.render(f'./{file_name}', view=True)

def choose_best_DT(df_features, df_responses):
    i = 1
    train_sum = 0
    test_sum = 0
    best_acc = 0
    best_train_x = None
    best_test_x = None
    best_val_x = None
    best_val_y = None
    best_test_y = None
    best_val_y = None
    while i < 11:
        train_X, X_val, test_X, train_y, y_val, test_y = val_split(
            df_features, df_responses, 0.6, 0.2)
        Tree = DecisionTree(
            train_X, train_y, column_first_names, MIN_LEAF_SIZE, MAX_HEIGHT)
        Tree.fit()
        Accuracy_Train = Tree.CALC_ACCURACY(train_X, train_y)
        Accuracy_Test = Tree.CALC_ACCURACY(test_X, test_y)
        print(f"split {i} traing completed")
        i = i+1
        print(f"Training accuracy is {Accuracy_Train}")
        print(f"Testing accuracy is {Accuracy_Test}")
        train_sum = train_sum + Accuracy_Train
        test_sum = test_sum + Accuracy_Test
        if Accuracy_Test <= best_acc:
            continue
        best_acc = Accuracy_Test
        best_train_x = train_X
        best_test_x = test_X
        best_train_y = train_y
        best_test_y = test_y
        best_val_x = X_val
        best_val_y = y_val
        best_tree=copy.deepcopy(Tree)
    train_sum = train_sum/10
    test_sum = test_sum/10
    
    print(f"\n\nmean train accuracy over 10 splits is {train_sum}")
    print(f"mean test accuracy over 10 splits is {test_sum}")
    return best_tree, best_train_x, best_test_x, best_val_x, best_train_y, best_test_y, best_val_y


def height_ablation(train_X, train_y,test_X,test_y ,X_val, y_val,column_first_names):

    num_node_list = []

    test_depth = []
    train_depth= []
    validation_depth= []

    for i in range(1, 11):
        print(f"Depth Check = {i}")
        tree3 = DecisionTree(
            train_X, train_y, column_first_names, MIN_LEAF_SIZE, i)
        tree3.fit()
        Xtest = tree3.predict(test_X)
        Xtrain = tree3.predict(train_X)
        val_   = tree3.predict(X_val)
        test_depth.append(calc_accuracy(test_y, Xtest))
        train_depth.append(calc_accuracy(train_y, Xtrain))
        validation_depth.append(calc_accuracy(y_val, val_))
        num_node_list.append(tree3.COUNT_NODES())

  
    plt.plot(range(1, 11), test_depth, label="Test")
    plt.plot(range(1, 11), train_depth, label="Train")
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.title("Train and Test Accuracy vs Max Depth")
    plt.show()

    Optimal_depth = 1+np.argmax(np.array(validation_depth))
    Best_tree = DecisionTree(
        train_X, train_y, column_first_names, MIN_LEAF_SIZE, Optimal_depth)
    Best_tree.fit()
    print(f"Optimal depth: {Optimal_depth}")
    print(f"Number of nodes: {Best_tree.COUNT_NODES()}")
    return Best_tree


def Igain_entropy_exp(df_features,df_responses,X_train_,y_train_,X_test_,y_test_,column_first_names):
    # training and testing and selecting best tree

    print("\nEntropy TREE" + "."*50)

    tree1 = DecisionTree(
        X_train_, y_train_, column_first_names, MIN_LEAF_SIZE, MAX_HEIGHT)
    tree1.fit()

    print("Completed Training")
    print("Training accuracy is ", tree1.CALC_ACCURACY(X_train_, y_train_))
    print("Testing accuracy is ", tree1.CALC_ACCURACY(X_test_, y_test_))
    

    print("\n"+"TREE over 10 random splits ENTROPY" + "."*50)
    best_dt, train_X, test_X, X_val, train_y, test_y,y_val = choose_best_DT(df_features, df_responses)
    print("\n"+ "BEST Test accuracy OVER 10 RANDOM SPLITS" + "."*50)
    print("Training accuracy:", best_dt.CALC_ACCURACY(train_X, train_y))
    print("Testing accuracy:", best_dt.CALC_ACCURACY(test_X, test_y))
    tree_to_gv(best_dt.root, column_first_names, "unpruned_dt.gv")


    print("\n"+ "DEPTH Vs TEST AND TRAIN Accuracy ENTROPY")

    BEST_TREE_HEIGHT = height_ablation(
        train_X, train_y, test_X, test_y,X_val,y_val, column_first_names)

    print("\n"+"PRUNING OPERATIONS"+"."*50)
    print("train_X:", train_X.shape)
    print("test_X:", test_X.shape)
    print("X_val:", X_val.shape)
    print("train_y:", train_y.shape)
    print("test_y:", test_y.shape)
    print("y_val:", y_val.shape)
    print("Unpruned best tree accuracies:")
    print("Training accuracy:", best_dt.CALC_ACCURACY(train_X, train_y))
    print("Testing accuracy:", best_dt.CALC_ACCURACY(test_X, test_y))

    # pruning
    best_dt.prune_post(train_X, train_y, X_val, y_val)
    print("\n"+"Pruning completed ...........")
    print("Training accuracy:", best_dt.calc_pruned_accuracy(train_X, train_y))
    print("Testing accuracy:", best_dt.calc_pruned_accuracy(test_X, test_y))
    print("Validation accuracy:", best_dt.calc_pruned_accuracy(X_val, y_val))
    tree_to_gv(best_dt.root_pruned, column_first_names, "pruned_dt.gv")
    tree_to_gv(BEST_TREE_HEIGHT.root, column_first_names,"best_unpruned_depth_optimal_entropy.gv")

def feature_names(path):
    '''
        Gets the column names from the given path
    '''
    data = pd.read_csv(path)
    return data.columns


def get_data_X_y(path):
    """
    Loads the X and y from the given path.
    :return: the x as X and y numpy arrays
    """
    from sklearn.preprocessing import LabelEncoder
    x = pd.read_csv(path)
    label = LabelEncoder()
    x['Gender'] = label.fit_transform(x['Gender'])
    x['Vehicle_Age'] = label.fit_transform(x['Vehicle_Age'])
    x['Vehicle_Damage'] = label.fit_transform(x['Vehicle_Damage'])
    X = x.drop(x.columns[-1], axis=1).to_numpy()
    y = x[x.columns[-1]].to_numpy()
    return X, y


def split(X, y, train_size):
    """
    Splits the x into training and test sets.
    """
  
    length = len(X)

    train_indices = int(train_size * length)
    X_train = X[:train_indices]
    X_test = X[train_indices:]
    y_train = y[:train_indices]
    y_test = y[train_indices:]
    return X_train, y_train, X_test,y_test


def val_split(X, y, train_size, val_size):
    '''
    Splits the x into training, validation and test sets.
    '''
    length = len(X)
    n_train = int(np.ceil(length*train_size))
    n_val = int(np.ceil(length*val_size))
    n_test = length - n_train - n_val

    X_train = X[n_train:]
    X_val = X[n_train:n_train+n_val]
    X_test = X[:n_test]
    y_train = y[n_train:]
    y_val = y[n_train:n_train+n_val]
    y_test = y[:n_test]
    return X_train, X_val, X_test, y_train, y_val, y_test


def purity_check(y):
    """
    Checks if the given array is pure.
    """
    return len(set(y)) == 1


def classify(y):
    """
    Classifies the array into a single class.
    """
    classes, counts = np.unique(y, return_counts=True)
    return classes[counts.argmax()]
   


def possible_break(X, type_arr):
    '''
        Calculates possible breaks for a given set of features 
    '''
    breaks = {}
    for col_idx in range(X.shape[1]):
        unique_vals = np.unique(X[:, col_idx])
        num_vals = np.unique(X[:, col_idx]).shape[0]

        type = type_arr[col_idx]
        if type == "cont":
            breaks[col_idx] = []
            i = 0
            for i in range(1, num_vals):
                current_value = unique_vals[i]
                previous_value = unique_vals[i - 1]
                potential_split = (current_value + previous_value) / 2
                breaks[col_idx].append(potential_split)
        elif num_vals > 1:
            breaks[col_idx] = unique_vals
    return breaks


def children_create(X, y, col_idx, col_val, type_arr):
    '''
        Creates the children of a dataset given split column and value
    '''
    y = y.reshape(-1, 1)
    X_n = np.hstack((X, y))
    relevant_column = X_n[:, col_idx]
   
    if type_arr[col_idx] == "cont":
        X_one = X_n[relevant_column <= col_val]
        X_two = X_n[relevant_column > col_val]
    else:
        X_one = X_n[relevant_column == col_val]
        X_two = X_n[relevant_column != col_val]

   
    Y_one = X_one[:, -1]
    Y_two = X_two[:, -1]
    X_one = X_one[:, :-1]
    X_two = X_two[:, :-1]

    return X_one, Y_one, X_two, Y_two


def entropy_cal(y):
    """
    Calculates the entropy of the given array.
    """
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return np.sum(probabilities * -np.log2(probabilities))

def infor_gain_cal(X, y, col_idx, col_val, type_arr):
    '''
        Calculates the information gain of a given split
    '''
    X_one, Y_one, X_two, Y_two = children_create(
        X, y, col_idx, col_val, type_arr)
    p = len(X_one) / len(X)
    return entropy_cal(y) - (p * entropy_cal(Y_one) + (1 - p) * entropy_cal(Y_two))

def get_best_split(X, y, type_arr):
    '''
        Calculates the best split for a given set of features
    '''
    best_col = -1
    best_val = -1
    best_gain = -10000
    breaks = possible_break(X, type_arr)
    for col_idx in breaks:
        for col_val in breaks[col_idx]:
            
            gain = infor_gain_cal(X, y, col_idx, col_val, type_arr)
            
            if gain > best_gain:
                best_col = col_idx
                best_val = col_val
                best_gain = gain
    return best_col, best_val


def feature_type(X, cont_thresh):
    '''
        Assigns the type of each feature based on the data
    '''
    type_arr = []
    for col_idx in range(X.shape[1]):
        type_val = X[:, col_idx][0]
        unique_vals = np.unique(X[:, col_idx])
        if len(unique_vals) < cont_thresh or isinstance(type_val, str):
            type_arr.append("discrete")
        else:
            type_arr.append("cont")
    return type_arr


def calc_accuracy(y_true, y_pred):
    '''
        Calculates the accuracy of the prediction
    '''
    return np.sum(y_pred == y_true) / len(y_pred)


def filter(X, y, col_idx, col_val, type_arr):

    y = y.reshape(-1, 1)
    X_n = np.hstack((X, y))
    relevant_column = X_n[:, col_idx]

    if type_arr[col_idx] == "cont":
        X_yes = X_n[relevant_column <= col_val]
        X_no = X_n[relevant_column > col_val]

    else:
        X_yes = X_n[relevant_column == col_val]
        X_no = X_n[relevant_column != col_val]

    Y_yes = X_yes[:, -1]
    Y_no = X_no[:, -1]
    X_yes = X_yes[:, :-1]
    X_no = X_no[:, :-1]
    return X_yes, Y_yes, X_no, Y_no


def check_node(X, y):
    return classify(y)

if __name__ == '__main__':
    
    MAX_HEIGHT = 10
    PATH_ = "Dataset_C.csv"
    MIN_LEAF_SIZE = 1
    print(f"MAX DEPTH = {MAX_HEIGHT}")
    print(f"DATA PATH = {PATH_}")

    print("PROCESSING" + "."*50)
    df_features, df_responses = get_data_X_y(PATH_)
    train_X, train_y, test_X, test_y = split(df_features, df_responses, TRAIN_VAL)
    print("X_train_size:", train_X.shape)
    print("y_train_size:", train_y.shape)
    print("X_test_size:", test_X.shape)
    print("y_test_size:", test_y.shape)
    column_first_names = feature_names(PATH_)

    Igain_entropy_exp(df_features, df_responses, train_X, train_y, test_X, test_y, column_first_names)

