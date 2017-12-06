from gensim.corpora import WikiCorpus, MmCorpus


class Wiki:
    def __init__(self, filename):
        self.wiki = WikiCorpus(filename)

    def __iter__(self):
        for article in self.wiki.get_texts():
            for sent in article:
                yield [word for word in sent.split('\s')]

    def save(self, filename):
        self.wiki.save(filename+'.dict')
        MmCorpus.serialize(filename+'.mm', self.wiki)


if __name__ == "__main__":
    import args
    wiki = Wiki(args.wiki)
    wiki.save('wiki_dict')
