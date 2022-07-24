from benji_transformer_play import *

if __name__ == '__main__':
    a = np.array([[-26.2533, -110.0013, -67.5635, -64.3901],
                  [-46.0843, -59.5493, -35.6727, -7.6016],
                  [-82.7170, -84.6684, -94.2187, -102.1358],
                  [-54.8882, -8.4623, -55.4405, -44.9618],
                  [-113.3286, -79.5341, -86.2449, -108.3712]])

    out = beam_search_matrix(a, n_beam=1)


    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True).to(device)

    data = datasets.load_dataset("universal_dependencies", "en_gum")

    scorer = Scorer2().to(device)
    scorer.load_state_dict(torch.load('runs/colab_07_21-14_26_16/save_epoch29.pt', map_location=device))

    training_data = [sequence for sequence in data['train'] if len(sequence['tokens']) < 40]

    val_data = [sequence for sequence in data['test'] if len(sequence['tokens']) < 40]


    input_data = val_data[:4]
    sentences_embedding, sentences_word_embeddings, targets = get_embeddings(input_data, tokenizer, model)
    i = 1
    word_embeddings, sentence_embedding, target = sentences_word_embeddings[i], sentences_embedding[i], targets[i]

    loss, pred, constr, mst_prob, mst_neg_log_prob, target_prob, target_neg_log_probs = do_train(word_embeddings,
                                                                                                 sentence_embedding,
                                                                                                 target, scorer)


    G1 = get_graphviz(input_data[i]['tokens'], constr[1:])
    print(input_data[0]['tokens'])
    G2 = get_graphviz(input_data[i]['tokens'], list(target.detach().numpy()))