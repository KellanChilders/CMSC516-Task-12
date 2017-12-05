from gensim.corpora import WikiCorpus, MmCorpus


def create_corpus(wiki_dump):
    wiki = WikiCorpus(wiki_dump)
    print('Saving')
    wiki.save('wiki_dict.dict')
    # MmCorpus.serialize('wiki_corpus.mm', wiki)


if __name__ == "__main__":
    import args
    create_corpus(args.wiki())
