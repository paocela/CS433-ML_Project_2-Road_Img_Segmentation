import numpy as np

def oversample(train_features, train_labels, index_of_1s, index_of_0s):
  index_of_0s = []
  index_of_1s = []
  for i in range(len(train_labels)):
      if train_labels[i][0] == 1:
          index_of_0s.append(i)
      else:
          index_of_1s.append(i)

  pos_features = train_features[index_of_1s, :, :, :]
  neg_features = train_features[index_of_0s, :, :, :]

  pos_labels = train_labels[index_of_1s]
  neg_labels = train_labels[index_of_0s]

  ids = np.arange(len(pos_features))
  choices = np.random.choice(ids, len(neg_features))

  res_pos_features = pos_features[choices, :, :, :]
  res_pos_labels = pos_labels[choices]

  resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
  resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)
  order = np.arange(len(resampled_labels))
  np.random.shuffle(order)
  resampled_features = resampled_features[order]
  resampled_labels = resampled_labels[order]

  resampled_features.shape
  
  return resampled_features, resampled_labels


def undersample(train_data, train_labels):
  c0 = 0  # bgrd
  c1 = 0  # road
  for i in range(len(train_labels)):
      if train_labels[i][0] == 1:
          c0 = c0 + 1
      else:
          c1 = c1 + 1

  min_c = min(c0, c1)
  idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
  idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
  new_indices = idx0[0:min_c] + idx1[0:min_c]
  new_train_data = train_data[new_indices, :, :, :]
  new_train_labels = train_labels[new_indices]
  return new_train_data, new_train_labels



# Split in train and test set
def train_test_split(train_data, train_labels, ratio, seed=1):
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(train_data)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    tr_data = train_data[index_tr]
    te_data = train_data[index_te]
    tr_labels = train_labels[index_tr]
    te_labels = train_labels[index_te]
    return tr_data, tr_labels, te_data, te_labels
