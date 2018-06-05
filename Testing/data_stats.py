

def report_raw_stats(data, keys, vocab):
    all_source_length = 0
    all_targets = []
    nb_source_sentences = 0
    all_nb_unk_source = 0
    all_nb_unk_targets = 0
    sample_without_unk = 0
    for k in keys:
        if k in data:
            source, refs = data[k]
            all_nb_unk_source += len([w for w in source.split(" ") if w not in vocab.word2index])
            all_nb_unk_targets += sum([len([w for w in ref.split(" ") if w not in vocab.word2index]) for ref in refs])
            all_source_length += len(source.split(" "))
            all_targets.append([len(t.split(" ")) for t in refs])
            nb_source_sentences += len(source.split(" . "))
            if sum([len([w for w in ref.split(" ") if w not in vocab.word2index]) for ref in refs]) == 0:
                sample_without_unk += 1

    print('Total', len([k for k in keys if k in data]))
    print('all_nb_unk_source', '\t', all_nb_unk_source / len([k for k in keys if k in data]))
    print('all_nb_unk_targets', '\t',all_nb_unk_targets / len([k for k in keys if k in data]))
    print('nb_source_sentences ', '\t',nb_source_sentences / len([k for k in keys if k in data]))
    print('all_source_length ', '\t',all_source_length / len([k for k in keys if k in data]))
    print('avg_ref_per_sent_length', '\t',
          sum([sum([t for t in target]) / len(target) for target in all_targets]) / len([k for k in keys if k in data]))

    print('avg_nb_ref_sents', '\t', sum([len(target) for target in all_targets]) / len([k for k in keys if k in data]))
    print('avg_total_ref_length',  '\t', sum([sum([t for t in target]) for target in all_targets]) / len([k for k in keys if k in data]))

    print('sample_without_unk', sample_without_unk / len([k for k in keys if k in data]))







