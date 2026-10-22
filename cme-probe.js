// CME probe: checks which public CME data pages answer from this machine, and what they return.
// Run: node cme-probe.js   (Node 18+). Prints a short report and saves probe-output.txt
'use strict';
const fs = require('fs');
const pad = (n) => String(n).padStart(2, '0');
// last weekday (CME publishes the previous trading day's settlements)
const d = new Date(Date.now() - 24 * 3600e3);
while ([0, 6].includes(d.getUTCDay())) d.setUTCDate(d.getUTCDate() - 1);
const ymd = `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}`;
const mdy = `${pad(d.getUTCMonth() + 1)}/${pad(d.getUTCDate())}/${d.getUTCFullYear()}`;
const B = 'https://www.cmegroup.com/CmeWS/mvc';
const URLS = [
  ['Futures volume/OI (6E)', `${B}/Volume/Details/F/58/${ymd}/P?tradeDate=${ymd}`],
  ['Futures settlements (6E)', `${B}/Settlements/Futures/Settlements/58/FUT?strategy=DEFAULT&tradeDate=${encodeURIComponent(mdy)}&pageSize=50`],
  ['Option series list (6E)', `${B}/Options/Categories/List/58/G?optionTypeFilter=`],
  ['Option volume/OI list (6E)', `${B}/Volume/Details/O/58/${ymd}/P?tradeDate=${ymd}`],
  ['Option settlements (6E, default)', `${B}/Settlements/Options/Settlements/58/OOF?tradeDate=${encodeURIComponent(mdy)}&pageSize=500`],
];
const HEAD = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36',
  Accept: 'application/json, text/plain, */*', Referer: 'https://www.cmegroup.com/markets/fx/g10/euro-fx.settlements.options.html' };
(async () => {
  let out = `CME probe ${new Date().toISOString()}  tradeDate=${ymd}\n`;
  for (const [name, url] of URLS) {
    let line;
    try {
      const r = await fetch(url, { headers: HEAD, signal: AbortSignal.timeout(20000) });
      const txt = await r.text();
      line = `\n=== ${name}\nHTTP ${r.status}  ${url}\n${txt.slice(0, 1200).replace(/\s+/g, ' ')}\n`;
    } catch (e) { line = `\n=== ${name}\nFAILED ${e.message}  ${url}\n`; }
    out += line; console.log(line);
  }
  fs.writeFileSync('probe-output.txt', out);
  console.log('\nSaved probe-output.txt');
})();
