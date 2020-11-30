from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from gensim.test.utils import datapath
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data/test_arxiv_plain.txt', help='Path to directory where the data is stored')
    parser.add_argument('--model-dir', default='../model', help='Path to directory where the model is stored')
    parser.add_argument('--train', default=True, help='True for train, False for test mode')
    parser.add_argument('--n_topic', default=20, help='Number of of topics')
    args = parser.parse_args()
    model_dir = './model/model'
    dict_dir = './model/dict.txt'

    if args.train == True:
        print('Reading texts')
        with open(args.data_dir) as f_in:
            texts = f_in.read().split('\n')
        del texts[-1]
        for i in tqdm(range(len(texts))):
            texts[i] = texts[i].split()

        print('Generating corpora')
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        dictionary.save_as_text(dict_dir)

        print('Loading model')
        lda = LdaModel(corpus, num_topics=args.n_topic)
        lda.save(model_dir)
    else:
        lda = LdaModel.load(model_dir, mmap='r')
        dictionary = Dictionary()
        dictionary.load_from_text(dict_dir)

    print('Processing results')
    topics = lda.print_topics()
    with open('./report.txt', 'w') as f_out:
        for topic_id, topic_pair in topics:
            print(topic_id, end = ': ', file=f_out)
            topic_words = topic_pair.split('"')[1::2]
            topic_words = list(map(int, topic_words))
            topic_words = [dictionary.get(word) for word in topic_words]
            print(topic_words, file=f_out)

if __name__ == "__main__":
    main()
