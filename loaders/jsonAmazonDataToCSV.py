from json import loads

with open('../data/Cell_Phones_and_Accessories_5.json') as f:
    with open('../data/cell_phones.tsv', 'w') as out_f:
        print('overall\treview text', file=out_f)
        for row in f:
            data = loads(row)
            print('{}\t{}'.format('pos' if int(data['overall']) > 3 else 'negative', data['reviewText']), file=out_f)
