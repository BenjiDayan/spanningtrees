from typing import List, Tuple

import numpy as np
import torch
import tqdm
from transformers import AutoTokenizer, AutoModel
import datasets

from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from spanningtrees.mst import MST

torch.manual_seed(42)
np.random.seed(42)


def get_word_idx(sent: str, word: str):
    return sent.split(" ").index(word)


def get_hidden_states(encoded, token_ids_word, model, layers,):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
       Select only those subword token outputs that belong to our word of interest
       and average them."""
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
    # Only select the tokens that constitute the requested word
    word_tokens_output = output[token_ids_word]

    return word_tokens_output.mean(dim=0)



layers = [-4, -3, -2, -1]
def get_bert_seq_layers(output, layers=None):
    """output: from bert. layers are the hidden states we want to extract.
    apparently summing -4: gives a good representation """
    layers = [-4, -3, -2, -1] if layers is None else layers
    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    return torch.stack([states[i] for i in layers]).sum(0).squeeze()


def get_word_embedding(encoded, word_idx, model, layers=None):
    # Use last four layers by default
    layers = [-4, -3, -2, -1] if layers is None else layers

    # sentence = "I like cookies ."
    # idx = get_word_idx(sent, "cookies")

    # which tokens correspond to our desired word idx
    # e.g. array([3, 4, 5, 6], dtype=int64)
    token_ids_word = np.where(np.array(encoded.word_ids()) == word_idx)[0]
    # word_embedding = get_word_vector(sent, idx, tokenizer, model, layers)

    # get the mean hidden state whatevers
    return get_hidden_states(encoded, token_ids_word, model, layers)


def sentence_to_word_nice(sentence, tokenizer, encoded=None):
    if encoded is None:
        encoded = tokenizer.encode_plus(sentence)
    word_ids = encoded.word_ids()
    word_ids = {word_id: encoded.word_to_chars(word_id) for word_id in word_ids if word_id is not None}
    word_ids = {word_id: (charspan, sentence[charspan.start:charspan.end]) for word_id, charspan in word_ids.items()}
    return word_ids


def match_words(word_list1, word_list2):
    """Tries to match up the second word list as a sub hierarchy of word_list1
    e.g. if [Eye's, -, tracking] in 2nd and [Eye, 's, -tracking] in 2nd
    return [(0,), (0,), (1, 2)], False
    False = all_fit is whether any tuple has length > 1, i.e. word_list2 isn't a partition refinement of word_list1.
    """
    # spans1, spans2 = word_list_to_spans(word_list1), word_list_to_spans(word_list2)
    spans1, spans2 = [np.cumsum(list(map(len, word_list))) for word_list in [word_list1, word_list2]]
    out = []
    if len(spans1) == []:
        out = [(-1,)] * len(spans2)
        return out, len(spans2) > 0

    n = 0
    i = spans1[n]
    for j in spans2:
        if n < len(spans1):
            inside = (n,)
        else:
            out.append((-1,))
            continue

        if j == i:  # we can go to next span in spans2, which is assumed to fall into the next span in span1, so incr n
            n += 1
            if n < len(spans1):  # o/w we don't draw i and it is anyway ignored
                i = spans1[n]
        while j > i:
            n += 1
            if n >= len(spans1):
                inside += (-1,)
                out.append(inside)
                break
            i = spans1[n]
            inside += (n,)
        else:  # if we didn't run out of spans1 goto next span in spans2
            out.append(inside)
            continue

    all_fit = not any(map(lambda x: len(x) > 1, out))
    return out, all_fit


input_size = 768
hidden_size = 500  # seems just to be a single hidden layer of size 500.
dropout=0.33

class Scorer(nn.Module):
    """Simple scorer. Takes head and dep word embeddings, runs them through a head and dep encoder, then does a
    bilinear transformation and sums the result - gives score for head word -> dep word."""
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.dep = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.bilinear = nn.Bilinear(in1_features=hidden_size, in2_features=hidden_size, out_features=1)
        self.extra_linear = nn.Linear(in_features=hidden_size, out_features=1, bias=False)

    def forward(self, word_vec_head, word_vec_dep, tanh=False):
        head_vec = self.head(word_vec_head)
        dep_vec = self.dep(word_vec_dep)
        out = self.bilinear(head_vec, dep_vec) + self.extra_linear(head_vec + dep_vec)
        out = out.sum(dim=-1)
        # trying this to make output smaller?
        if tanh:
            out = F.tanh(out)
        return out

# Like Scorer above but slightly different, produces a whole list of head -> dep scores for a list of word embeddings
class Scorer2(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.dep = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.bilinear = nn.Bilinear(in1_features=hidden_size, in2_features=hidden_size, out_features=1)
        self.extra_linear = nn.Linear(in_features=hidden_size, out_features=1, bias=False)

    def forward(self, word_embeddings, sentence_embedding=None, tanh=False):
        """word_embeddings: n x input_size where n is the number of words in the sentence

        return: nxn matrix if no sentence_embedding o/w n vector."""
        dep_vecs = self.dep(word_embeddings)

        if sentence_embedding is None:
            head_vecs = self.head(word_embeddings)
            n = word_embeddings.shape[0]

            dep_vecs2 = torch.tile(dep_vecs, (n, 1))  # n^2, hidden_size

            head_vecs2 = head_vecs.expand(n, *head_vecs.shape)  # n, n, hidden_size
            head_vecs2 = head_vecs2.transpose(0, 1)
            head_vecs2 = head_vecs2.reshape(-1, head_vecs.shape[-1])

            out = self.bilinear(head_vecs2, dep_vecs2) + self.extra_linear(head_vecs2 + dep_vecs2)
            out = out.sum(dim=-1)  # n^2
            out = out.reshape(n, n)

        else:  # TODO remove
            head_vec = self.head(sentence_embedding)
            out = self.bilinear(sentence_embedding, )
        # trying this to make output smaller?
        if tanh:
            out = F.tanh(out)
        return out


# class Softmax_Scorer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.scorer = Scorer()
#         self.root_embedding = nn.Sequential
#
#     def forward(self, word_vecs_head, word_vec_dep):
#         scores = torch.stack([self.scorer(word_vec_head, word_vec_dep) for word_vec_head in word_vecs_head])
#         return F.softmax(scores, dim=-1)

def laplacian(A: torch.Tensor):
    L = -A  # L_ij = -A_ij
    n = len(A)
    L[range(n), range(n)] = A.sum(dim=0) # L_jj = deg_in(v_j) = sum_i A_ij assuming A_jj = 0 I think.
    return L

def kooplacian(A: torch.Tensor, r: torch.Tensor):
    """
    A_ij is the exp(score(i -> j)) from word i to word j. The first word is the additional root object which will
    point to the "root" word, hence A_1r is exp(score(r)) = ρ_r
    :param A: (nxn) torch.Tensor
    :return: (nxn) torch.Tensor
    """
    L = laplacian(A)
    L[0] = r  # L_1j = ρ_j  is assumed to be the first row of A.

    return L



def get_embeddings(input_data, tokenizer, model):
    sentences, targets, word_lists1 = [], [], []
    for datum in input_data:
        word_lists1.append(datum['tokens'])  # used for matching BERT words to input data words
        sentences.append(datum['text'])
        targets.append(torch.Tensor(list(map(int, datum['head']))).long().to(device))

    # sentences: List[str] = [training_data[i]['text'] for i in data_i]
    batch_encoded = tokenizer(sentences, padding="longest", return_tensors="pt", truncation=True)
    for key in ['input_ids', 'token_type_ids', 'attention_mask']:
        batch_encoded[key] = batch_encoded[key].to(device)

    # replacing some tokens with unkown - regularisation?
    # p_unk_dropout = 0.25
    # stay_same = torch.rand(batch_encoded['input_ids'].shape) > p_unk_dropout
    # is_padding = batch_encoded['input_ids'] == tokenizer.pad_token_id
    # batch_encoded['input_ids'] = torch.where(torch.logical_or(stay_same, is_padding), batch_encoded['input_ids'],
    #                                       tokenizer.unk_token_id)

    encodeds = [tokenizer(sentence) for sentence in sentences]
    sentences_word_ids: List[dict] = [sentence_to_word_nice(sentence, tokenizer, encoded) for sentence, encoded
                                      in zip(sentences, encodeds)]

    all_fits = []
    sentences_word_id_tuples = []
    for word_list1, sentence_word_ids in zip(word_lists1, sentences_word_ids):
        word_list2 = [sentence_word_ids[i][1] for i in range(len(sentence_word_ids))]
        correspondence, all_fit = match_words(word_list1, word_list2)
        all_fits.append(all_fit)
        sentences_word_id_tuples.append(correspondence)

    all_fits = np.array(all_fits)


    with torch.no_grad():
        output = model(**batch_encoded)

    last_layers = get_bert_seq_layers(output)

    sentences_embedding = last_layers[:, 0, :].unsqueeze(dim=1)  # first token of each sequence embedding

    sentences_word_embeddings = []
    for sentence_i, (BERT_to_input_word_ids, encoded, all_fit) in enumerate(zip(sentences_word_id_tuples, encodeds, all_fits)):
        if not all_fit:
            sentences_word_embeddings.append([torch.Tensor([1.0]).to(device)])  # will be ignored later anyway
            continue

        sentences_word_embeddings.append([])
        BERT_token_word_ids = np.array(encoded.word_ids())
        # each tuple is one element, so e.g. [0, 0, 1, 2, 3, 4, 4, ...]. 1st 2 BERT words are word 0 of input and so on
        BERT_word_id_to_input_word_id = [x[0] for x in BERT_to_input_word_ids]
        BERT_word_ids_grouped = group_indices_by_num(BERT_word_id_to_input_word_id)
        BERT_word_id_tuple: Tuple[int]
        for BERT_word_id_tuple in BERT_word_ids_grouped:
            token_ids = np.where(
                np.logical_or.reduce([
                    BERT_token_word_ids == word_id
                for word_id in BERT_word_id_tuple])
            )[0]
            sentences_word_embeddings[-1].append(last_layers[sentence_i, token_ids].mean(dim=0))

    sentences_word_embeddings = [torch.stack(word_embeddings) for word_embeddings in sentences_word_embeddings]

    all_fits_indices = np.where(all_fits)[0]
    return sentences_embedding[all_fits_indices], [sentences_word_embeddings[i] for i in all_fits_indices], [targets[i] for i in all_fits_indices]


def group_indices_by_num(int_list):
    """[0, 0, 1, 2, 3, 4, 4, 5] -> [(0,1), (2,), (3,), (4,), (5,6), (7,)]"""
    grouped = []
    for i, num in enumerate(int_list):
        if len(grouped) < num+1:
            grouped.append((i,))
        else:
            grouped[-1] += (i,)
    return grouped


def get_partition(r_exp, exp_mat):

    #### Calculate partition function Z
    L1 = laplacian(exp_mat)
    L2 = kooplacian(exp_mat, r_exp)

    # Z1, Z2 should be equal - Z2 computation is just more efficient. See Koo paper
    Z1 = torch.tensor(0.).to(device)
    n = len(r_exp)
    for i in range(n):
        indices = list(range(n))
        indices.remove(i)
        Z1 += r_exp[i] * torch.det(L1[indices, :][:, indices])

    Z2 = torch.det(L2)

    return Z1, Z2


def get_mst(extended_mat, r, mat, Z, target):
    # #### Find maximum spanning tree, and its corresponding score
    # extended_mat = torch.concat([r.unsqueeze(dim=0), mat], dim=0)
    # extended_mat = torch.concat([torch.zeros((extended_mat.shape[0], 1)), extended_mat], dim=1)

    mst = MST(extended_mat.detach().cpu().numpy())
    constr = mst.mst(True)
    # constr looks like e.g. [-1, 4, 1, 0, 3]
    constr2 = constr[1:] - 1
    # target looks like e.g. [4, 1, 0, 3]
    target2 = target - 1
    mst_exp_score = torch.tensor(1.).to(device)

    # Anej
    mst_score_anej = torch.tensor(0.).to(device)

    def get_log_score(tree, r, mat):
        """tree: e.g. [3, 0, -1, 2] - word 3 is the root"""
        score = torch.tensor(0.).to(device)
        for dep_i, head_i in enumerate(tree):
            if head_i == -1:
                # mst_exp_score *= r_exp[dep_i]
                score += r[dep_i]
            else:
                # mst_exp_score *= exp_mat[head_i, dep_i]
                score += mat[head_i, dep_i]

        return score

    mst_neg_log_probs = - get_log_score(constr2, r, mat) + torch.log(Z)
    target_neg_log_probs = - get_log_score(target2, r, mat) + torch.log(Z)

    mst_prob = torch.exp(- mst_neg_log_probs)
    target_prob = torch.exp(- target_neg_log_probs)

    return constr, mst_prob, mst_neg_log_probs, target_prob, target_neg_log_probs




def do_train(word_embeddings, sentence_embedding, target, scorer):
    big_mat = scorer(torch.concat([sentence_embedding, word_embeddings]), tanh=False)

    mat = big_mat[1:, 1:]
    r = big_mat[0, 1:]

    # interestingly the grad for exp seems to depend on the output not input. Hence clone here, to avoid
    # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
    exp_mat = torch.exp(mat).clone()
    r_exp = torch.exp(r)

    n = len(r)
    exp_mat[range(n), range(n)] = 0.

    Z1, Z2 = get_partition(r_exp, exp_mat)

    #### Find maximum spanning tree, and its corresponding score
    extended_mat = torch.concat([r.unsqueeze(dim=0), mat], dim=0)
    extended_mat = torch.concat([torch.zeros((extended_mat.shape[0], 1)).to(device), extended_mat], dim=1)

    constr, mst_prob, mst_neg_log_probs, target_prob, target_neg_log_probs = get_mst(extended_mat, r, mat, Z2, target)

    # target = torch.tensor(list(map(int, a['head'])))
    # loss = F.cross_entropy(torch.concat([r.unsqueeze(dim=-1), mat], dim=1), torch.tensor(constr[1:], dtype=torch.long),
    #                 reduction="sum")

    # (N, C) where N is # of words and C is # of classes
    pred = torch.concat([r.unsqueeze(dim=-1), mat.T], dim=1)
    loss = F.cross_entropy(pred,
                           target,
                           reduction="sum")

    loss_anej = mst_neg_log_probs

    return loss, pred, constr, mst_prob, mst_neg_log_probs, target_prob, target_neg_log_probs


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main2(lr = 1e-3, n_epochs=100, n_batch=5, hist_weights_every=5, log_dir=None):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True).to(device)

    data = datasets.load_dataset("universal_dependencies", "en_gum")

    scorer = Scorer2().to(device)



    writer = SummaryWriter(log_dir=log_dir)

    # lr = 1e-4
    weight_decay = 0
    betas = (0.9, 0.9)
    eps=1e-12
    optimizer = torch.optim.Adam(scorer.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    # train_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    # my_dataloader = DataLoader(my_dataset)

    training_data = [sequence for sequence in data['train'] if len(sequence['tokens']) < 40]

    val_data = [sequence for sequence in data['test'] if len(sequence['tokens']) < 40]
    val_indices = np.arange(len(val_data))
    # n_epochs = 20
    # n_batch = 32
    # hist_weights_every = 5
    for epoch in range(n_epochs):
        train_indices = np.random.permutation(len(training_data))
        train_indices = [train_indices[n_batch * i:n_batch * (i + 1)] for i in range(len(train_indices) // n_batch)]
        loop = tqdm.tqdm(train_indices)
        for iter_i, data_i in enumerate(loop):
            input_data = [training_data[i] for i in data_i]

            # extract an embedding for each sentence, as well as embeddings for each word in the sentence.
            # some sentences (and corresponding targets) are filtered out if we cannot achieve a partition refinement
            # relation between input data words and the BERT tokens (actually the BERT words for simplification)
            sentences_embedding, sentences_word_embeddings, targets = get_embeddings(input_data, tokenizer, model)

            optimizer.zero_grad()
            losses = []
            mst_probs = []
            target_probs = []
            for word_embeddings, sentence_embedding, target in zip(sentences_word_embeddings, sentences_embedding, targets):

                loss, pred, constr, mst_prob, mst_neg_log_prob, target_prob, target_neg_log_probs = do_train(word_embeddings, sentence_embedding, target, scorer)

                # if mst_neg_log_prob != torch.inf and not mst_neg_log_prob.isnan():
                #     total_mst_neg_log_probs += mst_neg_log_prob
                mst_probs.append(mst_prob)
                target_probs.append(target_prob)

                if loss != torch.inf and not loss.isnan():
                    losses.append(loss)
                else:
                    print('Loss was inf/nan! Oh dear')

            losses, mst_probs, target_probs = map(torch.stack, [losses, mst_probs, target_probs])
            losses.sum().backward()

            # print(mst_log_probs)
            optimizer.step()
            overall_iter = epoch * len(train_indices) + iter_i
            writer.add_scalar("mean_cross_entropy_loss", losses.mean(), overall_iter)
            writer.add_scalar("mean_mst_prob", mst_probs.mean(), overall_iter)
            writer.add_scalar("target_prob", target_probs.mean(), overall_iter)
            # writer.add_scalar("mst_neg_log_probs", total_mst_neg_log_probs, overall_iter)
            # writer.add_scalar("cross_entropy_loss", total_loss, overall_iter)
            writer.add_scalar("batch_mean_sentence_length", np.mean([len(target) for target in targets]), overall_iter)
            writer.add_scalar("batch_num_sentences", len(targets), overall_iter)

            if iter_i % hist_weights_every == 0:
                writer.add_histogram('head.weight', scorer.head[0].weight, overall_iter)
                writer.add_histogram('head.bias', scorer.head[0].bias, overall_iter)
                writer.add_histogram('dep.weight', scorer.dep[0].weight, overall_iter)
                writer.add_histogram('dep.bias', scorer.dep[0].bias, overall_iter)
                writer.add_histogram('bilinear.weight', scorer.bilinear.weight, overall_iter)
                writer.add_histogram('bilinear.bias', scorer.bilinear.bias, overall_iter)

                input_indices = np.random.choice(val_indices, n_batch)
                input_data = [val_data[i] for i in input_indices]
                sentences_embedding, sentences_word_embeddings, targets = get_embeddings(input_data, tokenizer, model)

                losses = []
                mst_probs = []
                target_probs = []
                val_accs = []
                for word_embeddings, sentence_embedding, target in zip(sentences_word_embeddings, sentences_embedding,
                                                                       targets):

                    loss, pred, constr, mst_prob, mst_neg_log_prob, target_prob, target_neg_log_probs = do_train(word_embeddings, sentence_embedding,
                                                                              target, scorer)

                    pred_mst = torch.Tensor(constr[1:]).to(device)
                    acc = pred_mst == target
                    val_accs.append(acc.mean(dtype=torch.float))

                    # if mst_neg_log_prob != torch.inf and not mst_neg_log_prob.isnan():
                    #     total_mst_neg_log_probs += mst_neg_log_prob
                    mst_probs.append(mst_prob)
                    target_probs.append(target_prob)

                    if loss != torch.inf and not loss.isnan():
                        losses.append(loss)
                    else:
                        print('Loss was inf/nan! Oh dear')

                losses, mst_probs, target_probs = map(torch.stack, [losses, mst_probs, target_probs])

                writer.add_scalar("val_mean_cross_entropy_loss", losses.mean(), overall_iter)
                writer.add_scalar("val_mean_mst_prob", mst_probs.mean(), overall_iter)
                writer.add_scalar("val_target_prob", target_probs.mean(), overall_iter)
                # writer.add_scalar("val_mst_neg_log_probs", total_mst_neg_log_probs, overall_iter)
                # writer.add_scalar("val_cross_entropy_loss", total_loss, overall_iter)
                writer.add_scalar("val_mean_acc", torch.mean(torch.Tensor(val_accs)), overall_iter)

            loop.set_description(f"Epoch [{epoch}/{n_epochs}]")
            loop.set_postfix(loss=losses.mean().item())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--n_batch", default=5, type=int)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--log_dir")
    parser.add_argument("--hist_weights_every", default=5, type=int)
    args = parser.parse_args()
    print(vars(args))
    main2(**vars(args))

