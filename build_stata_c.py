import json
import pandas as pd

datadir = r'D:\Citypedia\Citypedia Project\Database Citypedia\HIES\Stata-data-1 (2)\Stata data'
cons = pd.read_stata(datadir + r'\sec_6a_consum_exp.dta', convert_categoricals=False)
cons = cons[cons['itc'] >= 10000].copy()
cons['itc6'] = cons['itc'].astype(int).astype(str).str.zfill(6)
unique_codes = sorted(list(cons['itc6'].unique()))

with open('recall_map.json', 'r') as f:
    mapping = json.load(f)

parts = {}
for code in unique_codes:
    if code.startswith(('127001', '127002', '127003', '127004', '127005')):
        continue
    part = mapping.get(code, 'Y')
    if code.startswith(('041', '042', '043')): part = 'Y'
    if code.startswith(('044', '045', '11')): part = 'M'
    if code.startswith(('06', '07', '08', '09', '10', '12')): part = 'Y'
    if code.startswith(('03', '05')): part = 'Y'
    parts[code] = part

f_codes = [c for c, p in parts.items() if p == 'F']
m_codes = [c for c, p in parts.items() if p == 'M']

out = open('stata_part_fixes.txt', 'w')
chunk_size = 9 # limits for inlist on strings is 10 (incl. the variable), so 9 parameters

for i in range(0, len(f_codes), chunk_size):
    chunks = f_codes[i:i+chunk_size]
    out.write('replace monthly_val = item_val * (365.25 / 14 / 12) if inlist(itc6, ' + ', '.join([f'"{c}"' for c in chunks]) + ')\n')

out.write('\n* Apply explicit PBS Monthly mapping\n')
for i in range(0, len(m_codes), chunk_size):
    chunks = m_codes[i:i+chunk_size]
    out.write('replace monthly_val = item_val if inlist(itc6, ' + ', '.join([f'"{c}"' for c in chunks]) + ')\n')

out.close()
