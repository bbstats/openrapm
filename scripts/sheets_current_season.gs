// Google Sheets: the season in progress, pulled from the site's ratings file.
// Paste into the sheet's Extensions > Apps Script, save, reload the sheet, then use the OpenRAPM menu.
// "Refresh now" pulls once; "Refresh every morning" sets one daily pull at 8 AM, after the 6 AM site update.

const URL = 'https://openrapm.com/data/ratings_list.json';
const TAB = 'Current season';
const HEAD = ['#', 'Player', 'Team', 'Offense', 'Defense', 'Total', 'WOSA', 'Poss.', 'Player ID'];

function onOpen() {
  SpreadsheetApp.getUi().createMenu('OpenRAPM')
    .addItem('Refresh now', 'refresh')
    .addItem('Refresh every morning', 'everyMorning')
    .addToUi();
}

function refresh() {
  const doc = JSON.parse(UrlFetchApp.fetch(URL).getContentText());
  const meta = doc.meta, season = String(meta.default_season), at = k => meta.fields.indexOf(k);
  const rows = doc.rows[season]
    .map(r => [r[at('n')], r[at('tm')], r[at('o')], r[at('d')], r[at('t')], r[at('w')], r[at('p')], r[at('id')]])
    .sort((a, b) => b[4] - a[4])
    .map((r, i) => [i + 1, ...r]);
  const book = SpreadsheetApp.getActive();
  const sheet = book.getSheetByName(TAB) || book.insertSheet(TAB);
  sheet.clearContents();
  sheet.getRange(1, 1, 1, HEAD.length).setValues([HEAD]);
  sheet.getRange(2, 1, rows.length, HEAD.length).setValues(rows);
  sheet.getRange('K1:L2').setValues([['Season', Number(season)], ['Through', (meta.through || {})[season] || meta.built]]);
  sheet.setFrozenRows(1);
  if (sheet.getFilter()) sheet.getFilter().remove();      // the filter grows with the player count
  sheet.getRange(1, 1, rows.length + 1, HEAD.length).createFilter();
}

function everyMorning() {
  ScriptApp.getProjectTriggers()
    .filter(t => t.getHandlerFunction() === 'refresh')
    .forEach(t => ScriptApp.deleteTrigger(t));
  ScriptApp.newTrigger('refresh').timeBased().everyDays(1).atHour(8).create();
}
