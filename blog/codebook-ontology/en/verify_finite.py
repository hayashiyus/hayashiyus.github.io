#!/usr/bin/env python3
"""Independent exhaustive check for the finite cancellation construction."""
from itertools import product
import json
from pathlib import Path

OUTCOMES=list(product((0,1),repeat=2))
ORIGINAL=((1,0),(0,1)); REPLACEMENT=((1,1),(0,0))
U={'U00':lambda x:x[0], 'U10':lambda x:x[1], 'U01':lambda x:1-x[0], 'U11':lambda x:1-x[1]}
def argmax(menu, utility): return max(menu, key=utility)
def transform(x, flip): return ((1-x[0],x[1]) if flip=='A' else (x[0],1-x[1]) if flip=='B' else x)
def diagnostic_choice(menu, utility, flip): return argmax(menu, lambda x:utility(transform(x,flip)))

def main():
    utilities={name:{'values':[fn(x) for x in OUTCOMES],
        'original_choice':list(argmax(ORIGINAL,fn)),
        'replacement_choice':list(argmax(REPLACEMENT,fn))} for name,fn in U.items()}
    repair_order=(('none','U11'),('target only','U01'),('preference only','U10'),('both','U00'))
    repairs=[]
    for label,name in repair_order:
        choice=argmax(ORIGINAL,U[name]); score=choice[0]
        repairs.append({'label':label,'utility':name,'original_choice':list(choice),'human_score':score})
    assert [r['human_score'] for r in repairs] == [1,0,0,1]
    expected_values={'U00':[0,0,1,1], 'U10':[0,1,0,1], 'U01':[1,1,0,0], 'U11':[1,0,1,0]}
    expected_choices={'U00':([1,0],[1,1]), 'U10':([0,1],[1,1]), 'U01':([0,1],[0,0]), 'U11':([1,0],[0,0])}
    for name, row in utilities.items():
        assert row['values'] == expected_values[name]
        assert (row['original_choice'],row['replacement_choice']) == expected_choices[name]
    assert all(U['U00'](x)==U['U11'](x) for x in ORIGINAL)
    diagnostics={flip:{name:list(diagnostic_choice(ORIGINAL,fn,flip)) for name,fn in U.items()} for flip in ('none','A','B')}
    assert diagnostics['A']['U00'] != diagnostics['none']['U00'] and diagnostics['A']['U11'] == diagnostics['none']['U11']
    assert diagnostics['B']['U11'] != diagnostics['none']['U11'] and diagnostics['B']['U00'] == diagnostics['none']['U00']
    for name in U:
        uses_a = name in ('U00','U01')
        assert (diagnostics['A'][name] != diagnostics['none'][name]) == uses_a
        assert (diagnostics['B'][name] != diagnostics['none'][name]) == (not uses_a)
    result={'outcomes':[list(x) for x in OUTCOMES], 'original_menu':[list(x) for x in ORIGINAL],
            'replacement_menu':[list(x) for x in REPLACEMENT], 'scorehuman':'x[0]',
            'utilities':utilities, 'repairs':repairs, 'diagnostics':diagnostics,
            'diagnostic_note':'Internal A/B flips are separate diagnostics; physical menu remains fixed.'}
    (Path(__file__).parent/'finite_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: utilities, menus, repair scores [1,0,0,1], and A/B diagnostics')
if __name__=='__main__': main()
