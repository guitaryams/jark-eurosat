"""
Check your submission file before you upload it.

    python check_submission.py my_submission.csv

It compares your file against sample_submission.csv, which lists every test id,
and tells you exactly what is wrong if anything is. It does not need the test
data and it cannot tell you your score, because it has no labels. It only
answers one question: will this file be accepted?

Run it every time. A file that fails here will be rejected after you upload it.
"""
import csv
import os
import sys
from collections import Counter

CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

HERE = os.path.dirname(os.path.abspath(__file__))


def read_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.reader(f))


def fail(msg, *extra):
    print('REJECTED')
    print('  ' + msg)
    for e in extra:
        print('  ' + e)
    sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sub_path = sys.argv[1]

    sample_path = os.path.join(HERE, 'sample_submission.csv')
    if not os.path.exists(sample_path):
        sys.exit('cannot find sample_submission.csv next to this script')
    want_ids = [r[0] for r in read_rows(sample_path)[1:]]

    rows = read_rows(sub_path)
    if not rows:
        fail('the file is empty.')

    header = [h.strip().lower() for h in rows[0]]

    # the single most common mistake: pandas wrote its row index as a column
    if len(header) == 3 and header[1:] == ['id', 'label']:
        fail('there is an extra column before "id". This is almost always pandas '
             'writing its row index.',
             'Fix: use  df.to_csv("my_submission.csv", index=False)')

    if len(header) != 2:
        fail('expected 2 columns, found %d: %r' % (len(header), rows[0]))
    if header != ['id', 'label']:
        fail('the first line must be exactly:  id,label',
             'Yours is: %s' % ','.join(rows[0]))

    body = [r for r in rows[1:] if any(c.strip() for c in r)]
    if not body:
        fail('there is a header but no data.')

    bad_width = [i for i, r in enumerate(body, 2) if len(r) != 2]
    if bad_width:
        fail('%d line(s) do not have exactly 2 fields, first at line %d.'
             % (len(bad_width), bad_width[0]),
             'A comma inside a class name will do this.')

    ids = [r[0].strip() for r in body]
    labels = [r[1].strip() for r in body]

    # label given as a number instead of a class name
    numeric = [l for l in labels if l.lstrip('-').isdigit()]
    if numeric:
        fail('the label column holds numbers, not class names. Found %r.' % numeric[0],
             'Your model predicts an index 0 to 9. Convert it with the classes array:',
             '    labels = [classes[i] for i in pred]')

    unknown = sorted(set(labels) - set(CLASSES))
    if unknown:
        near = []
        for u in unknown[:3]:
            squashed = u.lower().replace(' ', '').replace('_', '')
            for c in CLASSES:
                if c.lower() == squashed:
                    near.append('%r should be %r' % (u, c))
        fail('unknown class name(s): %s' % ', '.join(repr(u) for u in unknown[:5]),
             *(near or ['Valid names are: ' + ', '.join(CLASSES)]))

    dup = [i for i, n in Counter(ids).items() if n > 1]
    if dup:
        fail('%d id(s) appear more than once, for example %s.' % (len(dup), dup[0]))

    missing = set(want_ids) - set(ids)
    extra = set(ids) - set(want_ids)
    if missing:
        fail('%d test id(s) have no prediction, for example %s.'
             % (len(missing), sorted(missing)[0]),
             'Every one of the %d test ids must appear exactly once.' % len(want_ids))
    if extra:
        fail('%d id(s) are not in the test set, for example %s.'
             % (len(extra), sorted(extra)[0]),
             'Do not invent ids. Use the ones in sample_submission.csv.')

    counts = Counter(labels)
    print('ACCEPTED')
    print('  %d rows, all %d test ids present exactly once.' % (len(body), len(want_ids)))
    print('  Row order does not matter; ids are matched, not positions.')
    print()
    print('  how many of each class you predicted:')
    for c in CLASSES:
        n = counts.get(c, 0)
        flag = '   <- you never predict this class' if n == 0 else ''
        print('    %-22s %5d%s' % (c, n, flag))
    if any(counts.get(c, 0) == 0 for c in CLASSES):
        print()
        print('  A class you never predict scores zero for that class, and the')
        print('  score averages all ten classes equally. Worth a look.')


if __name__ == '__main__':
    main()
