"""Pod side (pod_launch.sh job dir): argv files and datasets.txt for the 12 ks bench chunk on gw-shared, plus a
256-template sub-bank for the smoke test. Usage: setup_job.py BANK.hdf STORE_DIR"""
import shlex, sys
import h5py
bank, sd = sys.argv[1:3]
S, E = 1380077568, 1380089856
argv = shlex.split(open('inspiral_argv_H1_000.txt').read())


def setopt(av, k, v):
    av = list(av)
    if k in av:
        i = av.index(k)
        if v is None: del av[i:i + 2]
        else: av[i + 1] = v
    elif v is not None: av += [k, v]
    return av


for k in ('--output', '--bank-file', '--approximant'):
    argv = setopt(argv, k, None)
open('argv_base.txt', 'w').write(shlex.join(argv))
for ifo in ('H1', 'L1'):
    av = setopt(setopt(argv, '--hdf-store', f'{sd}/{ifo}-{S}-{E}.hdf'), '--channel-name', f'{ifo}:GWOSC_STRAIN')
    open(f'argv_{ifo}.txt', 'w').write(shlex.join(av))
open('datasets.txt', 'w').write(''.join(f'{ifo} {S} {E} {sd}/{ifo}-{S}-{E}.hdf\n' for ifo in ('H1', 'L1')))
with h5py.File(bank) as f, h5py.File('bank_256.hdf', 'w') as g:
    for k in f:
        g[k] = f[k][:256]
    for k, v in f.attrs.items():
        g.attrs[k] = v
print('ok')
