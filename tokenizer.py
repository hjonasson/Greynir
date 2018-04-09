"""

    Reynir: Natural language processing for Icelandic

    Tokenizer module

    Copyright (C) 2017 Miðeind ehf.

       This program is free software: you can redistribute it and/or modify
       it under the terms of the GNU General Public License as published by
       the Free Software Foundation, either version 3 of the License, or
       (at your option) any later version.
       This program is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
       GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see http://www.gnu.org/licenses/.


    The function tokenize() consumes a text string and
    returns a generator of tokens. Each token is a tuple,
    typically having the form (type, word, meaning),
    where type is one of the constants specified in the
    TOK class, word is the original word found in the
    source text, and meaning is a list of tuples with
    potential interpretations of the word, as retrieved
    from the BIN database of word forms.

"""

from contextlib import closing
from collections import namedtuple, defaultdict

import re
import codecs
import datetime

from settings import Settings, StaticPhrases, Abbreviations, AmbigPhrases, DisallowedNames
from settings import NamePreferences, changedlocale
from bindb import BIN_Db, BIN_Meaning
from scraperdb import SessionContext, Entity


# Recognized punctuation

LEFT_PUNCTUATION = "([„‚«#$€<°"
RIGHT_PUNCTUATION = ".,:;)]!%?“»”’‛‘…>–"
CENTER_PUNCTUATION = '"*&+=@©|—'
NONE_PUNCTUATION = "-/'~‘\\"
PUNCTUATION = LEFT_PUNCTUATION + CENTER_PUNCTUATION + RIGHT_PUNCTUATION + NONE_PUNCTUATION

# Punctuation that ends a sentence
END_OF_SENTENCE = frozenset(['.', '?', '!', '[…]'])
# Punctuation symbols that may additionally occur at the end of a sentence
SENTENCE_FINISHERS = frozenset([')', ']', '“', '»', '”', '’', '"', '[…]'])
# Punctuation symbols that may occur inside words
PUNCT_INSIDE_WORD = frozenset(['.', "'", '‘', "´", "’"]) # Period and apostrophes

# Hyphens that are cast to '-' for parsing and then re-cast
# to normal hyphens, en or em dashes in final rendering
HYPHENS = "—–-"
HYPHEN = '-' # Normal hyphen

# Hyphens that may indicate composite words ('fjármála- og efnahagsráðuneyti')
COMPOSITE_HYPHENS = "–-"
COMPOSITE_HYPHEN = '–' # en dash

# Quotes that can be found
SQUOTES = "'‚‛‘"
DQUOTES = '"“„”'


CLOCK_WORD = "klukkan"
CLOCK_ABBREV = "kl"

# Prefixes that can be applied to adjectives with an intervening hyphen
ADJECTIVE_PREFIXES = frozenset(["hálf", "marg", "semí"])

# Person names that are not recognized at the start of sentences
NOT_NAME_AT_SENTENCE_START = { "Annar" }

# Punctuation types: left, center or right of word

TP_LEFT = 1   # Whitespace to the left
TP_CENTER = 2 # Whitespace to the left and right
TP_RIGHT = 3  # Whitespace to the right
TP_NONE = 4   # No whitespace
TP_WORD = 5   # Flexible whitespace depending on surroundings

# Matrix indicating correct spacing between tokens

TP_SPACE = (
    # Next token is:
    # LEFT    CENTER  RIGHT   NONE    WORD
    # Last token was TP_LEFT:
    ( False,  True,   False,  False,  False),
    # Last token was TP_CENTER:
    ( True,   True,   True,   True,   True),
    # Last token was TP_RIGHT:
    ( True,   True,   False,  False,  True),
    # Last token was TP_NONE:
    ( False,  True,   False,  False,  False),
    # Last token was TP_WORD:
    ( True,   True,   False,  False,  True)
)

# Numeric digits

DIGITS = frozenset([d for d in "0123456789"]) # Set of digit characters

# Set of all cases (nominative, accusative, dative, possessive)

ALL_CASES = frozenset(["nf", "þf", "þgf", "ef"])

# Month names and numbers

MONTHS = {
    "janúar": 1,
    "febrúar": 2,
    "mars": 3,
    "apríl": 4,
    "maí": 5,
    "júní": 6,
    "júlí": 7,
    "ágúst": 8,
    "september": 9,
    "október": 10,
    "nóvember": 11,
    "desember": 12
}

# Days of the month spelled out
DAYS_OF_MONTH = {
    "fyrsti": 1,
    "fyrsta": 1,
    "annar": 2,
    "annan": 2,
    "þriðji": 3,
    "þriðja": 3,
    "fjórði": 4,
    "fjórða": 4,
    "fimmti": 5,
    "fimmta": 5,
    "sjötti": 6,
    "sjötta": 6,
    "sjöundi": 7,
    "sjöunda": 7,
    "áttundi": 8,
    "áttunda": 8,
    "níundi": 9,
    "níunda": 9,
    "tíundi": 10,
    "tíunda": 10,
    "ellefti": 11,
    "ellefta": 11,
    "tólfti": 12,
    "tólfta": 12,
    "þrettándi": 13,
    "þrettánda": 13,
    "fjórtándi": 14,
    "fjórtánda": 14,
    "fimmtándi": 15,
    "fimmtánda": 15,
    "sextándi": 16,
    "sextánda": 16,
    "sautjándi": 17,
    "sautjánda": 17,
    "átjándi": 18,
    "átjánda": 18,
    "nítjándi": 19,
    "nítjánda": 19,
    "tuttugasti": 20,
    "tuttugasta": 20,
    "þrítugasti": 30,
    "þrítugasta": 30,
}

# Time of day expressions spelled out
CLOCK_NUMBERS = {
    "eitt": [1,0,0],
    "tvö": [2,0,0],
    "þrjú": [3,0,0],
    "fjögur": [4,0,0],
    "fimm": [5,0,0],
    "sex": [6,0,0],
    "sjö": [7,0,0],
    "átta": [8,0,0],
    "níu": [9,0,0],
    "tíu": [10,0,0],
    "ellefu": [11,0,0],
    "tólf": [12,0,0],
    "hálfeitt": [12,30,0],
    "hálftvö": [1,30,0],
    "hálfþrjú": [2,30,0],
    "hálffjögur": [3,30,0],
    "hálffimm": [4,30,0],
    "hálfsex": [5,30,0],
    "hálfsjö": [6,30,0],
    "hálfátta": [7,30,0],
    "hálfníu": [8,30,0],
    "hálftíu": [9,30,0],
    "hálfellefu": [10,30,0],
    "hálftólf": [11,30,0],
}

# Set of words only possible in temporal phrases
CLOCK_HALF = frozenset([
    "hálfeitt", 
    "hálftvö", 
    "hálfþrjú", 
    "hálffjögur",
    "hálffimm",
    "hálfsex",
    "hálfsjö",
    "hálfátta",
    "hálfníu",
    "hálftíu",
    "hálfellefu",
    "hálftólf"
])

# Set of word forms that are allowed to appear more than once in a row
ALLOWED_MULTIPLES = frozenset([
    "af",
    "auður",
    "að",
    "bannið",
    "bara",
    "bæði",
    "efni",
    "eftir",
    "eftir ",
    "eigi",
    "eigum",
    "eins",
    "ekki",
    "er",
    "er ",
    "falla",
    "fallið",
    "ferð",
    "festi",
    "flokkar",
    "flæði",
    "formið",
    "fram",
    "framan",
    "frá",
    "fylgi",
    "fyrir",
    "fyrir ",
    "fá",
    "gegn",
    "gerði",
    "getum",
    "hafa",
    "hafi",
    "hafið",
    "haft",
    "halla",
    "heim",
    "hekla",
    "heldur",
    "helga",
    "helgi",
    "hita",
    "hjá",
    "hjólum",
    "hlaupið",
    "hrætt",
    "hvort",
    "hæli",
    "inn ",
    "inni",
    "kanna",
    "kaupa",
    "kemba",
    "kira",
    "koma",
    "kæra",
    "lagi",
    "lagið",
    "leik",
    "leikur",
    "leið",
    "liðið",
    "lækna",
    "lögum",
    "löngu",
    "manni",
    "með",
    "milli",
    "minnst",
    "mun",
    "myndir",
    "málið",
    "móti",
    "mörkum",
    "neðan",
    "niðri",
    "niður",
    "niður ",
    "næst",
    "ofan",
    "opnir",
    "orðin",
    "rennur",
    "reynir",
    "riðlar",
    "riðli",
    "ráðum",
    "rétt",
    "safnið",
    "sem",
    "sett",
    "skipið",
    "skráðir",
    "spenna",
    "standa",
    "stofna",
    "streymi",
    "strokið",
    "stundum",
    "svala",
    "sæti",
    "sé",
    "sér",
    "síðan",
    "sótt",
    "sýna",
    "talið",
    "til",
    "tíma",
    "um",
    "undan",
    "undir",
    "upp",
    "upp ",
    "valda",
    "vanda",
    "var",
    "vega",
    "veikir",
    "vel",
    "velta",
    "vera",
    "verið",
    "vernda",
    "verða",
    "verði",
    "verður",
    "veður",
    "vikum",
    "við",
    "væri",
    "yfir",
    "yrði",
    "á",
    "á ",
    "átta",
    "í",
    "í ",
    "ó",
    "ómar",
    "úr",
    "út",
    "út ",
    "úti",
    "úti ",
    "þegar",
    "þjóna",
    ])

# Words incorrectly written as one word
NOT_COMPOUNDS = { 
    "afhverju" : ("af", "hverju"),
    "aftanfrá" : ("aftan", "frá"),
    "afturábak" : ("aftur", "á", "bak"),
    "afturí" : ("aftur", "í"),
    "afturúr" : ("aftur", "úr"),
    "afþví" : ("af", "því"),
    "afþvíað" : ("af", "því", "að"),
    "allajafna" : ("alla", "jafna"),
    "allajafnan" : ("alla", "jafnan"),
    "allrabest" : ("allra", "best"),
    "allrafyrst" : ("allra", "fyrst"),
    "allsekki" : ("alls", "ekki"),
    "allskonar" : ("alls", "konar"),
    "allskostar" : ("alls", "kostar"),
    "allskyns" : ("alls", "kyns"),
    "allsstaðar" : ("alls", "staðar"),
    "allstaðar" : ("alls", "staðar"),
    "alltsaman" : ("allt", "saman"),
    "alltíeinu" : ("allt", "í", "einu"),
    "alskonar" : ("alls", "konar"),
    "alskyns" : ("alls", "kyns"),
    "alstaðar" : ("alls", "staðar"),
    "annarhver" : ("annar", "hver"),
    "annarhvor" : ("annar", "hvor"),
    "annarskonar" : ("annars", "konar"),
    "annarslags" : ("annars", "lags"),
    "annarsstaðar" : ("annars", "staðar"),
    "annarstaðar" : ("annars", "staðar"),
    "annarsvegar" : ("annars", "vegar"),
    "annartveggja" : ("annar", "tveggja"),
    "annaðslagið" : ("annað", "slagið"),
    "austanfrá" : ("austan", "frá"),
    "austanmegin" : ("austan", "megin"),
    "austantil" : ("austan", "til"),
    "austureftir" : ("austur", "eftir"),
    "austurfrá" : ("austur", "frá"),
    "austurfyrir" : ("austur", "fyrir"),
    "bakatil" : ("baka", "til"),
    "báðumegin" : ("báðum", "megin"),
    "eftirað" : ("eftir", "að"),
    "eftirá" : ("eftir", "á"),
    "einhverjusinni" : ("einhverju", "sinni"),
    "einhverntíma" : ("einhvern", "tíma"),
    "einhverntímann" : ("einhvern", "tímann"),
    "einhvernveginn" : ("einhvern", "veginn"),
    "einhverskonar" : ("einhvers", "konar"),
    "einhversstaðar" : ("einhvers", "staðar"),
    "einhverstaðar" : ("einhvers", "staðar"),
    "einskisvirði" : ("einskis", "virði"),
    "einskonar" : ("eins", "konar"),
    "einsog" : ("eins", "og"),
    "einusinni" : ("einu", "sinni"),
    "eittsinn" : ("eitt", "sinn"),
    "endaþótt" : ("enda", "þótt"),
    "enganveginn" : ("engan", "veginn"),
    "ennfrekar" : ("enn", "frekar"),
    "ennfremur" : ("enn", "fremur"),
    "ennþá" : ("enn", "þá"),
    "fimmhundruð" : ("fimm", "hundruð"),
    "fimmtuhlutar" : ("fimmtu", "hlutar"),
    "fjórðuhlutar" : ("fjórðu", "hlutar"),
    "fjögurhundruð" : ("fjögur", "hundruð"),
    "framaf" : ("fram", "af"),
    "framanaf" : ("framan", "af"),
    "frameftir" : ("fram", "eftir"),
    "framhjá" : ("fram", "hjá"),
    "frammí" : ("frammi", "í"),
    "framundan" : ("fram", "undan"),
    "framundir" : ("fram", "undir"),
    "framvið" : ("fram", "við"),
    "framyfir" : ("fram", "yfir"),
    "framá" : ("fram", "á"),
    "framávið" : ("fram", "á", "við"),
    "framúr" : ("fram", "úr"),
    "fulltaf" : ("fullt", "af"),
    "fyrirfram" : ("fyrir", "fram"),
    "fyrren" : ("fyrr", "en"),
    "fyrripartur" : ("fyrr", "partur"),
    "heilshugar" : ("heils", "hugar"),
    "helduren" : ("heldur", "en"),
    "hinsvegar" : ("hins", "vegar"),
    "hinumegin" : ("hinum", "megin"),
    "hvarsem" : ("hvar", "sem"),
    "hvaðaner" : ("hvaðan", "er"),
    "hvaðansem" : ("hvaðan", "sem"),
    "hvaðeina" : ("hvað", "eina"),
    "hverjusinni" : ("hverju", "sinni"),
    "hverskonar" : ("hvers", "konar"),
    "hverskyns" : ("hvers", "kyns"),
    "hversvegna" : ("hvers", "vegna"),
    "hvertsem" : ("hvert", "sem"),
    "hvortannað" : ("hvort", "annað"),
    "hvorteðer" : ("hvort", "eð", "er"),
    "hvortveggja" : ("hvort", "tveggja"),
    "héreftir" : ("hér", "eftir"),
    "hérmeð" : ("hér", "með"),
    "hérnamegin" : ("hérna", "megin"),
    "hérumbil" : ("hér", "um", "bil"),
    "innanfrá" : ("innan", "frá"),
    "innanum" : ("innan", "um"),
    "inneftir" : ("inn", "eftir"),
    "innivið" : ("inni", "við"),
    "innvið" : ("inn", "við"),
    "inná" : ("inn", "á"),
    "innávið" : ("inn", "á", "við"),
    "inní" : ("inn", "í"),
    "innúr" : ("inn", "úr"),
    "lítilsháttar" : ("lítils", "háttar"),
    "margskonar" : ("margs", "konar"),
    "margskyns" : ("margs", "kyns"),
    "meirasegja" : ("meira", "að", "segja"),
    "meiraðsegja" : ("meira", "að", "segja"),
    "meiriháttar" : ("meiri", "háttar"),
    "meðþvíað" : ("með", "því", "að"),
    "mikilsháttar" : ("mikils", "háttar"),
    "minniháttar" : ("minni", "háttar"),
    "minnstakosti" : ("minnsta", "kosti"),
    "mörghundruð" : ("mörg", "hundruð"),
    "neinsstaðar" : ("neins", "staðar"),
    "neinstaðar" : ("neins", "staðar"),
    "niðreftir" : ("niður", "eftir"),
    "niðrá" : ("niður", "á"),
    "niðrí" : ("niður", "á"),
    "niðureftir" : ("niður", "eftir"),
    "niðurfrá" : ("niður", "frá"),
    "niðurfyrir" : ("niður", "fyrir"),
    "niðurá" : ("niður", "á"),
    "niðurávið" : ("niður", "á", "við"),
    "nokkrusinni" : ("nokkru", "sinni"),
    "nokkurntíma" : ("nokkurn", "tíma"),
    "nokkurntímann" : ("nokkurn", "tímann"),
    "nokkurnveginn" : ("nokkurn", "veginn"),
    "nokkurskonar" : ("nokkurs", "konar"),
    "nokkursstaðar" : ("nokkurs", "staðar"),
    "nokkurstaðar" : ("nokkurs", "staðar"),
    "norðanfrá" : ("norðan", "frá"),
    "norðanmegin" : ("norðan", "megin"),
    "norðantil" : ("norðan", "til"),
    "norðaustantil" : ("norðaustan", "til"),
    "norðureftir" : ("norður", "eftir"),
    "norðurfrá" : ("norður", "frá"),
    "norðurúr" : ("norður", "úr"),
    "norðvestantil" : ("norðvestan", "til"),
    "norðvesturtil" : ("norðvestur", "til"),
    "níuhundruð" : ("níu", "hundruð"),
    "núþegar" : ("nú", "þegar"),
    "ofanaf" : ("ofan", "af"),
    "ofaná" : ("ofan", "á"),
    "ofaní" : ("ofan", "í"),
    "ofanúr" : ("ofan", "úr"),
    "oní" : ("ofan", "í"),
    "réttumegin" : ("réttum", "megin"),
    "réttummegin" : ("réttum", "megin"),
    "samskonar" : ("sams", "konar"),
    "seinnipartur" : ("seinni", "partur"),
    "semsagt" : ("sem", "sagt"),
    "sexhundruð" : ("sex", "hundruð"),
    "sigrihrósandi" : ("sigri", "hrósandi"),
    "sjöhundruð" : ("sjö", "hundruð"),
    "sjöttuhlutar" : ("sjöttu", "hlutar"),
    "smámsaman" : ("smám", "saman"),
    "sumsstaðar" : ("sums", "staðar"),
    "sumstaðar" : ("sums", "staðar"),
    "sunnanað" : ("sunnan", "að"),
    "sunnanmegin" : ("sunnan", "megin"),
    "sunnantil" : ("sunnan", "til"),
    "sunnanvið" : ("sunnan", "við"),
    "suðaustantil" : ("suðaustan", "til"),
    "suðuraf" : ("suður", "af"),
    "suðureftir" : ("suður", "eftir"),
    "suðurfrá" : ("suður", "frá"),
    "suðurfyrir" : ("suður", "fyrir"),
    "suðurí" : ("suður", "í"),
    "suðvestantil" : ("suðvestan", "til"),
    "svoað" : ("svo", "að"),
    "svokallaður" : ("svo", "kallaður"),
    "svosem" : ("svo", "sem"),
    "svosemeins" : ("svo", "sem", "eins"),
    "svotil" : ("svo", "til"),
    "tilbaka" : ("til", "baka"),
    "tilþessað" : ("til", "þess", "að"),
    "tvennskonar" : ("tvenns", "konar"),
    "tvöhundruð" : ("tvö", "hundruð"),
    "tvöþúsund" : ("tvö", "þúsund"),
    "umfram" : ("um", "fram"),
    "undanúr" : ("undan", "úr"),
    "undireins" : ("undir", "eins"),
    "uppaf" : ("upp", "af"),
    "uppað" : ("upp", "að"),
    "uppeftir" : ("upp", "eftir"),
    "uppfrá" : ("upp", "frá"),
    "uppundir" : ("upp", "undir"),
    "uppá" : ("upp", "á"),
    "uppávið" : ("upp", "á", "við"),
    "uppí" : ("upp", "í"),
    "uppúr" : ("upp", "úr"),
    "utanaf" : ("utan", "af"),
    "utanað" : ("utan", "að"),
    "utanfrá" : ("utan", "frá"),
    "utanmeð" : ("utan", "með"),
    "utanum" : ("utan", "um"),
    "utanundir" : ("utan", "undir"),
    "utanvið" : ("utan", "við"),
    "utaná" : ("utan", "á"),
    "vegnaþess" : ("vegna", "þess"),
    "vestantil" : ("vestan", "til"),
    "vestureftir" : ("vestur", "eftir"),
    "vesturyfir" : ("vestur", "yfir"),
    "vesturúr" : ("vestur", "úr"),
    "vitlausumegin" : ("vitlausum", "megin"),
    "viðkemur" : ("við", "kemur"),
    "viðkom" : ("við", "kom"),
    "viðkæmi" : ("við", "kæmi"),
    "viðkæmum" : ("við", "kæmum"),
    "víðsfjarri" : ("víðs", "fjarri"),
    "víðsvegar" : ("víðs", "vegar"),
    "yfirum" : ("yfir", "um"),
    "ámeðal" : ("á", "meðal"),
    "ámilli" : ("á", "milli"),
    "áttahundruð" : ("átta", "hundruð"),
    "áðuren" : ("áður", "en"),
    "öðruhverju" : ("öðru", "hverju"),
    "öðruhvoru" : ("öðru", "hvoru"),
    "öðrumegin" : ("öðrum", "megin"),
    "úrþvíað" : ("úr", "því", "að"),
    "útaf" : ("út", "af"),
    "útfrá" : ("út", "frá"),
    "útfyrir" : ("út", "fyrir"),
    "útifyrir" : ("út", "fyrir"),
    "útivið" : ("út", "við"),
    "útundan" : ("út", "undan"),
    "útvið" : ("út", "við"),
    "útá" : ("út", "á"),
    "útávið" : ("út", "á", "við"),
    "útí" : ("út", "í"),
    "útúr" : ("út", "úr"),
    "ýmiskonar" : ("ýmiss", "konar"),
    "ýmisskonar" : ("ýmiss", "konar"),
    "þangaðsem" : ("þangað", "sem"),
    "þarafleiðandi" : ("þar", "af", "leiðandi"),
    "þaraðauki" : ("þar", "að", "auki"),
    "þareð" : ("þar", "eð"),
    "þarmeð" : ("þar", "með"),
    "þarsem" : ("þar", "sem"),
    "þarsíðasta" : ("þar", "síðasta"),
    "þarsíðustu" : ("þar", "síðustu"),
    "þartilgerður" : ("þar", "til", "gerður"),
    "þeimegin" : ("þeim", "megin"),
    "þeimmegin" : ("þeim", "megin"),
    "þessháttar" : ("þess", "háttar"),
    "þesskonar" : ("þess", "konar"),
    "þesskyns" : ("þess", "kyns"),
    "þessvegna" : ("þess", "vegna"),
    "þriðjuhlutar" : ("þriðju", "hlutar"),
    "þrjúhundruð" : ("þrjú", "hundruð"),
    "þrjúþúsund" : ("þrjú", "þúsund"),
    "þvíað" : ("því", "að"),
    "þvínæst" : ("því", "næst"),
    "þínmegin" : ("þín", "megin"),
    "þóað" : ("þó", "að"),    
    }

SPLIT_COMPOUNDS = {
    ("afbragðs", "fagur") : "afbragðsfagur",
    ("afbragðs", "góður") : "afbragðsgóður",
    ("afbragðs", "maður") : "afbragðsmaður",
    ("afburða", "árangur") : "afburðaárangur",
    ("aftaka", "veður") : "aftakaveður",
    ("al", "góður") : "algóður",
    ("all", "góður") : "allgóður",
    ("allsherjar", "atkvæðagreiðsla") : "allsherjaratkvæðagreiðsla",
    ("allsherjar", "breyting") : "allsherjarbreyting",
    ("allsherjar", "neyðarútkall") : "allsherjarneyðarútkall",
    ("and", "stæðingur") : "andstæðingur",
    ("auka", "herbergi") : "aukaherbergi",
    ("auð", "sveipur") : "auðsveipur",
    ("aðal", "inngangur") : "aðalinngangur",
    ("aðaldyra", "megin") : "aðaldyramegin",
    ("bakborðs", "megin") : "bakborðsmegin",
    ("bakdyra", "megin") : "bakdyramegin",
    ("blæja", "logn") : "blæjalogn",
    ("brekku", "megin") : "brekkumegin",
    ("bílstjóra", "megin") : "bílstjóramegin",
    ("einskis", "verður") : "einskisverður",
    ("endur", "úthluta") : "endurúthluta",
    ("farþega", "megin") : "farþegamegin",
    ("fjölda", "margir") : "fjöldamargir",
    ("for", "maður") : "formaður",
    ("forkunnar", "fagir") : "forkunnarfagur",
    ("frum", "stæður") : "frumstæður",
    ("full", "mikill") : "fullmikill",
    ("furðu", "góður") : "furðugóður",
    ("gagn", "stæður") : "gagnstæður",
    ("gegn", "drepa") : "gegndrepa",
    ("ger", "breyta") : "gerbreyta",
    ("gjalda", "megin") : "gjaldamegin",
    ("gjör", "breyta") : "gjörbreyta",
    ("heildar", "staða") : "heildarstaða",
    ("hlé", "megin") : "hlémegin",
    ("hálf", "undarlegur") : "hálfundarlegur",
    ("hálfs", "mánaðarlega") : "hálfsmánaðarlega",
    ("hálftíma", "gangur") : "hálftímagangur",
    ("innvortis", "blæðing") : "innvortisblæðing",
    ("jafn", "framt") : "jafnframt",
    ("jafn", "lyndur") : "jafnlyndur",
    ("jafn", "vægi") : "jafnvægi",
    ("karla", "megin") : "karlamegin",
    ("klukkustundar", "frestur") : "klukkustundarfrestur",
    ("kring", "um") : "kringum",
    ("kvenna", "megin") : "kvennamegin",
    ("lang", "stærstur") : "langstærstur",
    ("langtíma", "aukaverkun") : "langtímaaukaverkun",
    ("langtíma", "lán") : "langtímalán",
    ("langtíma", "markmið") : "langtímamarkmið",
    ("langtíma", "skuld") : "langtímaskuld",
    ("langtíma", "sparnaður") : "langtímasparnaður",
    ("langtíma", "spá") : "langtímaspá",
    ("langtíma", "stefnumörkun") : "langtímastefnumörkun",
    ("langtíma", "þróun") : "langtímaþróun",
    ("lágmarks", "aldur") : "lágmarksaldur",
    ("lágmarks", "fjöldi") : "lágmarksfjöldi",
    ("lágmarks", "gjald") : "lágmarksgjald",
    ("lágmarks", "kurteisi") : "lágmarkskurteisi",
    ("lágmarks", "menntun") : "lágmarksmenntun",
    ("lágmarks", "stærð") : "lágmarksstærð",
    ("lágmarks", "áhætta") : "lágmarksáhætta",
    ("lítils", "verður") : "lítilsverður",
    ("marg", "oft") : "margoft",
    ("megin", "atriði") : "meginatriði",
    ("megin", "forsenda") : "meginforsenda",
    ("megin", "land") : "meginland",
    ("megin", "markmið") : "meginmarkmið",
    ("megin", "orsök") : "meginorsök",
    ("megin", "regla") : "meginregla",
    ("megin", "tilgangur") : "megintilgangur",
    ("megin", "uppistaða") : "meginuppistaða ",
    ("megin", "viðfangsefni") : "meginviðfangsefni",
    ("megin", "ágreiningur") : "meginágreiningur",
    ("megin", "ákvörðun") : "meginákvörðun",
    ("megin", "áveitukerfi") : "megináveitukerfi",
    ("mest", "allt") : "mestallt",
    ("mest", "allur") : "mestallur",
    ("meðal", "aðgengi") : "meðalaðgengi",
    ("meðal", "biðtími") : "meðalbiðtími",
    ("meðal", "ævilengd") : "meðalævilengd",
    ("mis", "bjóða") : "misbjóða",
    ("mis", "breiður") : "misbreiður",
    ("mis", "heppnaður") : "misheppnaður",
    ("mis", "lengi") : "mislengi",
    ("mis", "mikið") : "mismikið",
    ("mis", "stíga") : "misstíga",
    ("miðlungs", "beiskja") : "miðlungsbeiskja",
    ("myndar", "drengur") : "myndardrengur",
    ("næst", "bestur") : "næstbestur",
    ("næst", "komandi") : "næstkomandi",
    ("næst", "síðastur") : "næstsíðastur",
    ("næst", "verstur") : "næstverstur",
    ("sam", "skeyti") : "samskeyti",
    ("saman", "stendur") : "samanstendur",
    ("sjávar", "megin") : "sjávarmegin",
    ("skammtíma", "skuld") : "skammtímaskuld",
    ("skammtíma", "vistun") : "skammtímavistun",
    ("svo", "kallaður") : "svokallaður",
    ("sér", "framboð") : "sérframboð",
    ("sér", "herbergi") : "sérherbergi",
    ("sér", "inngangur") : "sérinngangur",
    ("sér", "kennari") : "sérkennari",
    ("sér", "staða") : "sérstaða",
    ("sér", "stæði") : "sérstæði",
    ("sér", "vitringur") : "sérvitringur",
    ("sér", "íslenskur") : "séríslenskur",
    ("sér", "þekking") : "sérþekking",
    ("sér", "þvottahús") : "sérþvottahús",
    ("sí", "felldur") : "sífelldur",
    ("sólar", "megin") : "sólarmegin",
    ("tor", "læs") : "torlæs",
    ("undra", "góður") : "undragóður",
    ("uppáhalds", "bragðtegund") : "uppáhaldsbragðtegund",
    ("uppáhalds", "fag") : "uppáhaldsfag",
    ("van", "megnugur") : "vanmegnugur",
    ("van", "virða") : "vanvirða",
    ("vel", "ferð") : "velferð",
    ("vel", "kominn") : "velkominn",
    ("vel", "megun") : "velmegun",
    ("vel", "vild") : "velvild",
    ("ágætis", "maður") : "ágætismaður",
    ("áratuga", "reynsla") : "áratugareynsla",
    ("áratuga", "skeið") : "áratugaskeið",
    ("óhemju", "illa") : "óhemjuilla",
    ("óhemju", "vandaður") : "óhemjuvandaður",
    ("óskapa", "hiti") : "óskapahiti",
    ("óvenju", "góður") : "óvenjugóður",
    ("önd", "verður") : "öndverður",
    ("ör", "magna") : "örmagna",
    ("úrvals", "hveiti") : "úrvalshveiti",
    # Split into 3 words
    #("heils", "dags", "starf") : "heilsdagsstarf",
    #("heils", "árs", "vegur") : "heilsársvegur",
    #("hálfs", "dags", "starf") : "hálfsdagsstarf",
    #("marg", "um", "talaður") : "margumtalaður",
    #("sama", "sem", "merki") : "samasemmerki",
    #("því", "um", "líkt") : "þvíumlíkt",

}


# Incorrectly written ordinals
ORDINAL_ERRORS = {
    "1sti" : "fyrsti",
    "1sta" : "fyrsta",
    "1stu" : "fyrstu",
    "3ji" : "þriðji",
    "3ja" : "þriðja",
    "3ju" : "þriðju",
    "4ði" : "fjórði",
    "4ða" : "fjórða",
    "4ðu" : "fjórðu",
    "5ti" : "fimmti",
    "5ta" : "fimmta",
    "5tu" : "fimmtu",
    "2svar" : "tvisvar",
    "3svar" : "þrisvar",
    "2ja" : "tveggja",
    "3ja" : "þriggja",
    "4ra" : "fjögurra"
}

# A = Area
# T = Time
# L = Length
# C = Temperature
# W = Weight
# V = Volume
SI_UNITS = {
    "m²" : "A",
    "fm" : "A",
    "cm²" : "A",
    "cm³" : "V",
    "ltr" : "V",
    "dl" : "V",
    "cl" : "V",
    "m³" : "V",
    "°C" : "C",
    "gr" : "W",
    "kg" : "W",
    "mg" : "W",
    "μg" : "W",
    "km" : "L",
    "mm" : "L",
    "cm" : "L",
    "sm" : "L",
}
# Handling of Roman numerals

RE_ROMAN_NUMERAL = re.compile(r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")

ROMAN_NUMERAL_MAP = tuple(zip(
    (1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1),
    ('M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I')
))

def roman_to_int(s):
    """ Quick and dirty conversion of an already validated Roman numeral to integer """
    # Adapted from http://code.activestate.com/recipes/81611-roman-numerals/
    i = result = 0
    for integer, numeral in ROMAN_NUMERAL_MAP:
        while s[i:i + len(numeral)] == numeral:
            result += integer
            i += len(numeral)
    assert i == len(s)
    return result

# Named tuple for person names, including case and gender

PersonName = namedtuple('PersonName', ['name', 'gender', 'case'])

# Named tuple for tokens

Tok = namedtuple('Tok', ['kind', 'txt', 'val', 'error'], )


def correct_spaces(s):
    """ Split and re-compose a string with correct spacing between tokens"""
    r = []
    last = TP_NONE
    for w in s.split():
        if len(w) > 1:
            this = TP_WORD
        elif w in LEFT_PUNCTUATION:
            this = TP_LEFT
        elif w in RIGHT_PUNCTUATION:
            this = TP_RIGHT
        elif w in NONE_PUNCTUATION:
            this = TP_NONE
        elif w in CENTER_PUNCTUATION:
            this = TP_CENTER
        else:
            this = TP_WORD
        if TP_SPACE[last - 1][this - 1] and r:
            r.append(" " + w)
        else:
            r.append(w)
        last = this
    return "".join(r)


# Token types

class TOK:

    # Note: Keep the following in sync with token identifiers in main.js

    PUNCTUATION = 1
    TIME = 2
    DATE = 3
    YEAR = 4
    NUMBER = 5
    WORD = 6
    TELNO = 7
    PERCENT = 8
    URL = 9
    ORDINAL = 10
    TIMESTAMP = 11
    CURRENCY = 12
    AMOUNT = 13
    PERSON = 14
    EMAIL = 15
    ENTITY = 16
    UNKNOWN = 17
    DATEABS = 18
    DATEREL = 19
    TIMESTAMPABS = 20
    TIMESTAMPREL = 21
    MEASUREMENT = 22

    P_BEGIN = 10001 # Paragraph begin
    P_END = 10002 # Paragraph end

    S_BEGIN = 11001 # Sentence begin
    S_END = 11002 # Sentence end

    END = frozenset((P_END, S_END))
    TEXT = frozenset((WORD, PERSON, ENTITY))
    TEXT_EXCL_PERSON = frozenset((WORD, ENTITY))

    # Token descriptive names

    descr = {
        PUNCTUATION: "PUNCTUATION",
        TIME: "TIME",
        TIMESTAMP: "TIMESTAMP",
        TIMESTAMPABS: "TIMESTAMPABS",
        TIMESTAMPREL: "TIMESTAMPREL",
        DATE: "DATE",
        DATEABS: "DATEABS",
        DATEREL: "DATEREL",
        YEAR: "YEAR",
        NUMBER: "NUMBER",
        CURRENCY: "CURRENCY",
        AMOUNT: "AMOUNT",
        MEASUREMENT: "MEASUREMENT",
        PERSON: "PERSON",
        WORD: "WORD",
        TELNO: "TELNO",
        PERCENT: "PERCENT",
        URL: "URL",
        EMAIL: "EMAIL",
        ORDINAL: "ORDINAL",
        ENTITY: "ENTITY",
        UNKNOWN: "UNKNOWN",
        P_BEGIN: "BEGIN PARA",
        P_END: "END PARA",
        S_BEGIN: "BEGIN SENT",
        S_END: "END SENT"
    }

    # Token constructors
    @staticmethod
    def Punctuation(w, error=[]):
        tp = TP_CENTER # Default punctuation type
        if w:
            if w[0] in LEFT_PUNCTUATION:
                tp = TP_LEFT
            elif w[0] in RIGHT_PUNCTUATION:
                tp = TP_RIGHT
            elif w[0] in NONE_PUNCTUATION:
                tp = TP_NONE
        return Tok(TOK.PUNCTUATION, w, tp, error)

    @staticmethod
    def Time(w, h, m, s, error=[]):
        return Tok(TOK.TIME, w, (h, m, s), error)

    @staticmethod
    def Date(w, y, m, d, error=[]):
        return Tok(TOK.DATE, w, (y, m, d), error)

    @staticmethod
    def Dateabs(w, y, m, d, error=[]):
        return Tok(TOK.DATEABS, w, (y, m, d), error)

    @staticmethod
    def Daterel(w, y, m, d, error=[]):
        return Tok(TOK.DATEREL, w, (y, m, d), error)

    @staticmethod
    def Timestamp(w, y, mo, d, h, m, s, error=[]):
        return Tok(TOK.TIMESTAMP, w, (y, mo, d, h, m, s), error)
   
    @staticmethod
    def Timestampabs(w, y, mo, d, h, m, s, error=[]):
        return Tok(TOK.TIMESTAMPABS, w, (y, mo, d, h, m, s), error)
  
    @staticmethod
    def Timestamprel(w, y, mo, d, h, m, s, error=[]):
        return Tok(TOK.TIMESTAMPREL, w, (y, mo, d, h, m, s), error)

    @staticmethod
    def Year(w, n, error=[]):
        return Tok(TOK.YEAR, w, n, error)

    @staticmethod
    def Telno(w, error=[]):
        return Tok(TOK.TELNO, w, None, error)

    @staticmethod
    def Email(w, error=[]):
        return Tok(TOK.EMAIL, w, None, error)

    @staticmethod
    def Number(w, n, cases=None, genders=None, error=[]):
        """ cases is a list of possible cases for this number
            (if it was originally stated in words) """
        return Tok(TOK.NUMBER, w, (n, cases, genders), error)

    @staticmethod
    def Currency(w, iso, cases=None, genders=None, error=[]):
        """ cases is a list of possible cases for this currency name
            (if it was originally stated in words, i.e. not abbreviated) """
        return Tok(TOK.CURRENCY, w, (iso, cases, genders), error)

    @staticmethod
    def Amount(w, iso, n, cases=None, genders=None, error=[]):
        """ cases is a list of possible cases for this amount
            (if it was originally stated in words) """
        return Tok(TOK.AMOUNT, w, (n, iso, cases, genders), error)

    @staticmethod
    def Measurement(w, si, n, error=[]):
        return Tok(TOK.MEASUREMENT, w, (si, n), error)

    @staticmethod
    def Percent(w, n, cases=None, genders=None, error=[]):
        return Tok(TOK.PERCENT, w, (n, cases, genders), error)

    @staticmethod
    def Ordinal(w, n, error=[]):
        return Tok(TOK.ORDINAL, w, n, error)

    @staticmethod
    def Url(w, error=[]):
        return Tok(TOK.URL, w, None, error)

    @staticmethod
    def Word(w, m, error=[]):
        """ m is a list of BIN_Meaning tuples fetched from the BÍN database """
        return Tok(TOK.WORD, w, m, error)

    @staticmethod
    def Unknown(w, error=[]):
        return Tok(TOK.UNKNOWN, w, None, error)

    @staticmethod
    def Person(w, m, error=[]):
        """ m is a list of PersonName tuples: (name, gender, case) """
        return Tok(TOK.PERSON, w, m, error)

    @staticmethod
    def Entity(w, definitions, cases=None, genders=None, error=[]):
        return Tok(TOK.ENTITY, w, (definitions, cases, genders), error)

    @staticmethod
    def Begin_Paragraph():
        return Tok(TOK.P_BEGIN, None, None, error=[])

    @staticmethod
    def End_Paragraph():
        return Tok(TOK.P_END, None, None, error=[])

    @staticmethod
    def Begin_Sentence(num_parses = 0, err_index = None):
        return Tok(TOK.S_BEGIN, None, (num_parses, err_index), error=[])

    @staticmethod
    def End_Sentence():
        return Tok(TOK.S_END, None, None, error=[])


def is_valid_date(y, m, d):
    """ Returns True if y, m, d is a valid date """
    if (1776 <= y <= 2100) and (1 <= m <= 12) and (1 <= d <= 31):
        try:
            datetime.datetime(year = y, month = m, day = d)
            return True
        except ValueError:
            pass
    return False


def parse_digits(w):
    """ Parse a raw token starting with a digit """
    s = re.match(r'\d{1,2}:\d\d:\d\d', w)
    if s:
        # Looks like a 24-hour clock, H:M:S
        w = s.group()
        p = w.split(':')
        h = int(p[0])
        m = int(p[1])
        sec = int(p[2])
        if (0 <= h < 24) and (0 <= m < 60) and (0 <= sec < 60):
            return TOK.Time(w, h, m, sec, error=[]), s.end()
    s = re.match(r'\d{1,2}:\d\d', w)
    if s:
        # Looks like a 24-hour clock, H:M
        w = s.group()
        p = w.split(':')
        h = int(p[0])
        m = int(p[1])
        if (0 <= h < 24) and (0 <= m < 60):
            return TOK.Time(w, h, m, 0, error=[]), s.end()
    s = re.match(r'\d{1,2}\.\d{1,2}\.\d{2,4}', w) or re.match(r'\d{1,2}/\d{1,2}/\d{2,4}', w)
    if s:
        # Looks like a date
        w = s.group()
        if '/' in w:
            p = w.split('/')
        else:
            p = w.split('.')
        y = int(p[2])
        # noinspection PyAugmentAssignment
        if y <= 99:
            y = 2000 + y
        m = int(p[1])
        d = int(p[0])
        if m > 12 >= d:
            # Probably wrong way around
            m, d = d, m
        if is_valid_date(y, m, d):
            return TOK.Date(w, y, m, d, error=[]), s.end()
    s = re.match(r'\d+(\.\d\d\d)*,\d+', w)
    if s:
        # Real number formatted with decimal comma and possibly thousands separator
        # (we need to check this before checking integers)
        w = s.group()
        n = re.sub(r'\.', '', w) # Eliminate thousands separators
        n = re.sub(r',', '.', n) # Convert decimal comma to point
        return TOK.Number(w, float(n), error=[]), s.end()
    s = re.match(r'\d+(\.\d\d\d)+', w)
    if s:
        # Integer with a '.' thousands separator
        # (we need to check this before checking dd.mm dates)
        w = s.group()
        n = re.sub(r'\.', '', w) # Eliminate thousands separators
        return TOK.Number(w, int(n), error=[]), s.end()
    s = re.match(r'\d{1,2}/\d{1,2}', w)
    if s and (s.end() >= len(w) or w[s.end()] not in DIGITS):
        # Looks like a date (and not something like 10/2007)
        w = s.group()
        p = w.split('/')
        m = int(p[1])
        d = int(p[0])
        if p[0][0] != '0' and p[1][0] != '0' and ((d <= 5 and m <= 6) or (d == 1 and m <= 10)):
            # This is probably a fraction, not a date
            # (1/2, 1/3, 1/4, 1/5, 1/6, 2/3, 2/5, 5/6 etc.)
            # Return a number
            return TOK.Number(w, float(d) / m, error=[]), s.end()
        if m > 12 >= d:
            # Date is probably wrong way around
            m, d = d, m
        if (1 <= d <= 31) and (1 <= m <= 12):
            # Looks like a (roughly) valid date
            return TOK.Date(w, 0, m, d, error=[]), s.end()
    s = re.match(r'\d\d\d\d$', w) or re.match(r'\d\d\d\d[^\d]', w)
    if s:
        n = int(w[0:4])
        if 1776 <= n <= 2100:
            # Looks like a year
            return TOK.Year(w[0:4], n, error=[]), 4
    s = re.match(r'\d\d\d-\d\d\d\d', w)
    if s:
        # Looks like a telephone number
        return TOK.Telno(s.group(), error=[]), s.end()
    s = re.match(r'\d\d\d\d\d\d\d', w)
    if s:
        # Looks like a telephone number
        return TOK.Telno(s.group()[:3] +"-" + s.group()[3:], error=[]), s.end()

    s = re.match(r'\d+\.\d+(\.\d+)+', w)
    if s:
        # Some kind of ordinal chapter number: 2.5.1 etc.
        # (we need to check this before numbers with decimal points)
        w = s.group()
        n = re.sub(r'\.', '', w) # Eliminate dots, 2.5.1 -> 251
        return TOK.Ordinal(w, int(n), error=[]), s.end()
    s = re.match(r'\d+(,\d\d\d)*\.\d+', w)
    if s:
        # Real number, possibly with a thousands separator and decimal comma/point
        w = s.group()
        n = re.sub(r',', '', w) # Eliminate thousands separators
        w = re.sub(r'\.', ',', w)   # Change decimal point separator to a comma, GrammCorr 1N
        return TOK.Number(w, float(n), error=[]), s.end()
    s = re.match(r'\d+(,\d\d\d)*', w)
    if s:
        # Integer, possibly with a ',' thousands separator
        w = s.group()
        n = re.sub(r',', '', w) # Eliminate thousands separators
        return TOK.Number(w, int(n), error=[]), s.end()
    # Strange thing
    return TOK.Unknown(w, error=[]), len(w)


def parse_tokens(txt):
    """ Generator that parses contiguous text into a stream of tokens """
    rough = txt.split()
    QM = False # Check if quotation marks on both ends of word    
    for w in rough:
        QM = False
        # Handle each sequence of non-whitespace characters
        if w.isalpha() or w in SI_UNITS:
            # Shortcut for most common case: pure word
            yield TOK.Word(w, None, error=[])
            continue

        # More complex case of mixed punctuation, letters and numbers
        
        if len(w) > 2 and w[0] in DQUOTES and w[-1] in DQUOTES:
            # Convert to matching Icelandic quotes
            QM = True
            yield TOK.Punctuation('„', error=[])
            if w[1:-1].isalpha():
                yield TOK.Word(w[1:-1], None, error=[])
                yield TOK.Punctuation('“', error=[])
                QM = False
                continue
            else:
                w = w[1:-1] + '“'
        if len(w) > 2 and w[0] in SQUOTES and w[-1] in SQUOTES:
            # Convert to matching Icelandic quotes
            QM = True
            yield TOK.Punctuation('‚', error=[1])
            if w[1:-1].isalpha():
                yield TOK.Word(w[1:-1], None, error=[])
                yield TOK.Punctuation('‘', error=[1])
                QM = False
                continue
            else:
                w = w[1:-1] + '‘'
        
        if len(w) > 1 and w[0] == '"':
            # Convert simple quotes to proper opening quotes
            yield TOK.Punctuation('„', error=[1])
            w = w[1:]

        while w:
            # Punctuation
            ate = False
            while w and w[0] in PUNCTUATION:
                ate = True
                if w.startswith("[...]"):
                    yield TOK.Punctuation("[…]", error=[])
                    w = w[5:]
                elif w.startswith("[…]"):
                    yield TOK.Punctuation("[…]", error=[])
                    w = w[3:]
                elif w.startswith("..."):
                    # Treat ellipsis as one piece of punctuation
                    yield TOK.Punctuation("…", error=[])
                    w = w[3:]
                elif w == ",,":
                    # Was at the end of a word or by itself, should be ",". GrammCorr 1K
                    yield TOK.Punctuation(',', error=[2])  
                    w = w[2:]
                elif w.startswith(",,"):
                    # Probably an idiot trying to type opening double quotes with commas
                    yield TOK.Punctuation('„', error=[1])
                    w = w[2:]
                elif len(w) == 2 and (w == "[[" or w == "]]"):
                    # Begin or end paragraph marker
                    if w == "[[":
                        yield TOK.Begin_Paragraph()
                    else:
                        yield TOK.End_Paragraph()
                    w = w[2:]
                elif w[0] in HYPHENS:
                    # Represent all hyphens the same way
                    yield TOK.Punctuation(HYPHEN)
                    w = w[1:]
                    # Any sequence of hyphens is treated as a single hyphen
                    while w and w[0] in HYPHENS:
                        w = w[1:]
                elif w == '”':
                    w = '“'
                    continue
                elif w == "'":
                    # Left with a single quote, convert to proper closing quote
                    w = "‘"
                    continue
                else:
                    yield TOK.Punctuation(w[0], error=[])
                    w = w[1:]
                if w == '"':
                    # We're left with a simple double quote: Convert to proper closing quote
                    w = '“'
                    continue

            if w and '@' in w:
                # Check for valid e-mail
                # Note: we don't allow double quotes (simple or closing ones) in e-mails here
                # even though they're technically allowed according to the RFCs
                s = re.match(r"[^@\s]+@[^@\s]+(\.[^@\s\.,/:;\"”]+)+", w)
                if s:
                    ate = True
                    yield TOK.Email(s.group(), error=[])
                    w = w[s.end():]
            # Numbers or other stuff starting with a digit
            if w and w[0] in DIGITS:
                if w in ORDINAL_ERRORS: # GrammCorr 1P
                    w = ORDINAL_ERRORS[w]
                    continue
                else:
                    ate = True
                    t, eaten = parse_digits(w)
                    yield t
                    # Continue where the digits parser left off
                    w = w[eaten:]
            if w and w.startswith("http://") or w.startswith("https://") or w.startswith("www"):
                # Handle URL: cut RIGHT_PUNCTUATION characters off its end,
                # even though many of them are actually allowed according to
                # the IETF RFC
                endp = ""
                while w and w[-1] in RIGHT_PUNCTUATION:
                    endp = w[-1] + endp
                    w = w[:-1]
                yield TOK.Url(w, error=[])
                ate = True
                w = endp

            # Alphabetic characters
            if w and w[0].isalpha():
                ate = True
                i = 1
                lw = len(w)
                while i < lw and (w[i].isalpha() or (w[i] in PUNCT_INSIDE_WORD and (i+1 == lw or w[i+1].isalpha()))):
                    # We allow dots to occur inside words in the case of
                    # abbreviations; also apostrophes are allowed within words and at the end
                    # (O'Malley, Mary's, it's, childrens', O‘Donnell)
                    i += 1
                # Make a special check for the occasional erroneous source text case where sentences
                # run together over a period without a space: 'sjávarútvegi.Það'
                a = w.split('.')
                if len(a) == 2 and a[0] and a[0][0].islower() and a[1] and a[1][0].isupper():
                    # We have a lowercase word immediately followed by a period and an uppercase word
                    yield TOK.Word(a[0], None, error=[])
                    yield TOK.Punctuation('.', error=[])
                    yield TOK.Word(a[1], None, error=[3])
                    w = None
                else:
                    while w[i-1] == '.':
                        # Don't eat periods at the end of words
                        i -= 1
                    yield TOK.Word(w[0:i], None, error=[])
                    w = w[i:]
                    if w and w[0] in COMPOSITE_HYPHENS:
                        # This is a hyphen or en dash directly appended to a word:
                        # might be a continuation ('fjármála- og efnahagsráðuneyti')
                        # Yield a special hyphen as a marker
                        yield TOK.Punctuation(COMPOSITE_HYPHEN, error=[])
                        w = w[1:]
                    if QM:
                        if w[:-1].isalpha():
                            yield TOK.Word(w[:-1], None, error=[])
                            if w[-1] in SQUOTES:
                                yield TOK.Punctuation('‘', error=[])
                            elif q[-1] in DQUOTES:
                                yield TOK.Punctuation('“', error=[])
                            else:
                                pass
                            QM = False
            if not ate:
                # Ensure that we eat everything, even unknown stuff
                yield TOK.Unknown(w[0], error=[])
                w = w[1:]
            # We have eaten something from the front of the raw token.
            # Check whether we're left with a simple double quote,
            # in which case we convert it to a proper closing double quote
            if w and w[0] == '"':
                w = '“' + w[1:]

def parse_particles(token_stream):
    """ Parse a stream of tokens looking for 'particles'
        (simple token pairs and abbreviations) and making substitutions """

    def is_abbr_with_period(txt):
        """ Return True if the given token text is an abbreviation when followed by a period """
        if '.' in txt:
            # There is already a period in it: must be an abbreviation
            return True
        if txt in Abbreviations.SINGLES:
            # The token's literal text is defined as an abbreviation followed by a single period
            return True
        if txt.lower() in Abbreviations.SINGLES:
            # The token is in upper or mixed case:
            # We allow it as an abbreviation unless the exact form (most often uppercase)
            # is an abbreviation that doesn't require a period (i.e. isn't in SINGLES).
            # This applies for instance to DR which means "Danmark's Radio" instead of "doktor" (dr.)
            return txt not in Abbreviations.DICT
        return False

    token = None
    try:

        # Maintain a one-token lookahead
        token = next(token_stream)
        while True:
            next_token = next(token_stream)
            # Make the lookahead checks we're interested in

            clock = False

            # Check for $[number]
            if token.kind == TOK.PUNCTUATION and token.txt == '$' and \
                next_token.kind == TOK.NUMBER:

                token = TOK.Amount(token.txt + next_token.txt, "USD", next_token.val[0], error=[token.error, next_token.error]) # Unknown gender
                next_token = next(token_stream)

            # Check for €[number]
            if token.kind == TOK.PUNCTUATION and token.txt == '€' and \
                next_token.kind == TOK.NUMBER:

                token = TOK.Amount(token.txt + next_token.txt, "EUR", next_token.val[0], error=[token.error, next_token.error]) # Unknown gender
                next_token = next(token_stream)

            # Coalesce abbreviations ending with a period into a single
            # abbreviation token
            if next_token.kind == TOK.PUNCTUATION and next_token.txt == '.':
                if token.kind == TOK.WORD and token.txt[-1] != '.' and is_abbr_with_period(token.txt):
                    # Abbreviation ending with period: make a special token for it
                    # and advance the input stream

                    clock = token.txt.lower() == CLOCK_ABBREV
                    follow_token = next(token_stream)
                    abbrev = token.txt + "."

                    # Check whether we might be at the end of a sentence, i.e.
                    # the following token is an end-of-sentence or end-of-paragraph,
                    # or uppercase (and not a month name misspelled in upper case).

                    if abbrev in Abbreviations.NAME_FINISHERS:
                        # For name finishers (such as 'próf.') we don't consider a
                        # following person name as an indicator of an end-of-sentence
                        # !!! BUG: This does not work as intended because person names
                        # !!! have not been recognized at this phase in the token pipeline.
                        test_set = TOK.TEXT_EXCL_PERSON
                    else:
                        test_set = TOK.TEXT

                    finish = ((follow_token.kind in TOK.END) or
                        (follow_token.kind in test_set and follow_token.txt[0].isupper() and
                        not follow_token.txt.lower() in MONTHS))

                    if finish:
                        # Potentially at the end of a sentence
                        if abbrev in Abbreviations.FINISHERS:
                            # We see this as an abbreviation even if the next sentence seems
                            # to be starting just after it.
                            # Yield the abbreviation without a trailing dot,
                            # and then an 'extra' period token to end the current sentence.
                            token = TOK.Word("[" + token.txt + "]", None, error=[])
                            yield token
                            token = next_token
                        elif abbrev in Abbreviations.NOT_FINISHERS:
                            # This is an abbreviation that we don't interpret as such
                            # if it's at the end of a sentence ('dags.', 'próf.', 'mín.')
                            yield token
                            token = next_token
                        else:
                            # Substitute the abbreviation and eat the period
                            token = TOK.Word("[" + token.txt + ".]", None, error=[])
                    else:
                        # 'Regular' abbreviation in the middle of a sentence:
                        # swallow the period and yield the abbreviation as a single token
                        token = TOK.Word("[" + token.txt + ".]", None, error=[])

                    next_token = follow_token

            # Coalesce 'klukkan'/[kl.] + time or number into a time
            if next_token.kind == TOK.TIME or next_token.kind == TOK.NUMBER:
                if clock or (token.kind == TOK.WORD and token.txt.lower() == CLOCK_WORD):
                    # Match: coalesce and step to next token
                    if next_token.kind == TOK.NUMBER:
                        token = TOK.Time(CLOCK_ABBREV + ". " + next_token.txt, next_token.val[0], 0, 0, error=compound_error([token.error, next_token.error]))
                    else:
                        token = TOK.Time(CLOCK_ABBREV + ". " + next_token.txt,
                            next_token.val[0], next_token.val[1], next_token.val[2], error=compound_error([token.error, next_token.error]))
                    next_token = next(token_stream)

            # Coalesce 'klukkan/kl. átta/hálfátta' into a time
            if next_token.txt in CLOCK_NUMBERS and (clock or (token.kind == TOK.WORD and token.txt.lower() == CLOCK_WORD)):
                # Match: coalesce and step to next token
                token = TOK.Time(CLOCK_ABBREV + ". " + next_token.txt, *CLOCK_NUMBERS[next_token.txt], error=compound_error([token.error, next_token.error]))
                next_token = next(token_stream)
            # Words like 'hálftólf' only used in temporal expressions so can stand alone
            if token.txt in CLOCK_HALF:
                token = TOK.Time(token.txt, *CLOCK_NUMBERS[token.txt], error=token.error)

            # Coalesce 'árið' + [year|number] into year
            if (token.kind == TOK.WORD and (token.txt == "árið" or token.txt == "ársins" or token.txt == "árinu")) and (next_token.kind == TOK.YEAR or next_token.kind == TOK.NUMBER):
                token = TOK.Year(token.txt + " " + next_token.txt, next_token.txt, error=compound_error([token.error, next_token.error]))
                next_token = next(token_stream)

            # Coalesce [year|number] + ['e.Kr.'|'f.Kr.'] into year
            if token.kind == TOK.YEAR or (token.kind == TOK.NUMBER):
                val = int(token.val) if token.kind == TOK.YEAR else token.val[0] if token.kind == TOK.NUMBER else 0
                if next_token.txt == "f.Kr":
                    token = TOK.Year(token.txt + " " + next_token.txt, -int(val), error=compound_error([token.error, next_token.error]))
                    next_token = next(token_stream)
                elif next_token.txt == "e.Kr":
                    token = TOK.Year(token.txt + " " + next_token.txt, val, error=compound_error([token.error, next_token.error]))
                    next_token = next(token_stream)

            # Coalesce percentages into a single token
            if next_token.kind == TOK.PUNCTUATION and next_token.txt == '%':
                if token.kind == TOK.NUMBER:
                    # Percentage: convert to a percentage token
                    # In this case, there are no cases and no gender
                    token = TOK.Percent(token.txt + '%', token.val[0], error=token.error)
                    next_token = next(token_stream)

            # Coalesce ordinals (1. = first, 2. = second...) into a single token
            if next_token.kind == TOK.PUNCTUATION and next_token.txt == '.':
                if (token.kind == TOK.NUMBER and not ('.' in token.txt or ',' in token.txt)) or \
                    (token.kind == TOK.WORD and RE_ROMAN_NUMERAL.match(token.txt)):
                    # Ordinal, i.e. whole number or Roman numeral followed by period: convert to an ordinal token
                    follow_token = next(token_stream)
                    if follow_token.kind in (TOK.S_END, TOK.P_END) or \
                        (follow_token.kind == TOK.PUNCTUATION and follow_token.txt in {'„', '"'}) or \
                        (follow_token.kind == TOK.WORD and follow_token.txt[0].isupper() and
                        follow_token.txt.lower() not in MONTHS):
                        # Next token is a sentence or paragraph end,
                        # or opening quotes,
                        # or an uppercase word (and not a month name misspelled in upper case):
                        # fall back from assuming that this is an ordinal
                        yield token # Yield the number or Roman numeral
                        token = next_token # The period
                        next_token = follow_token # The following (uppercase) word or sentence end
                    else:
                        # OK: replace the number/Roman numeral and the period with an ordinal token
                        num = token.val[0] if token.kind == TOK.NUMBER else roman_to_int(token.txt)
                        token = TOK.Ordinal(token.txt + '.', num, error=token.error)
                        # Continue with the following word
                        next_token = follow_token

            if token.kind == TOK.NUMBER and next_token.txt in SI_UNITS:
                token = TOK.Measurement(token.txt + " " + next_token.txt, SI_UNITS[next_token.txt], token.val, error=compound_error([token.error, next_token.error]))
                next_token = next(token_stream)
            # Yield the current token and advance to the lookahead
            yield token
            token = next_token

    except StopIteration:
        # Final token (previous lookahead)
        if token:
            yield token


def parse_sentences(token_stream):
    """ Parse a stream of tokens looking for sentences, i.e. substreams within
        blocks delimited by sentence finishers (periods, question marks,
        exclamation marks, etc.) """
    in_sentence = False
    token = None
    tok_begin_sentence = TOK.Begin_Sentence()
    tok_end_sentence = TOK.End_Sentence()

    try:

        # Maintain a one-token lookahead
        token = next(token_stream)
        while True:
            next_token = next(token_stream)

            if token.kind == TOK.P_BEGIN or token.kind == TOK.P_END:
                # Block start or end: finish the current sentence, if any
                if in_sentence:
                    yield tok_end_sentence
                    in_sentence = False
                if token.kind == TOK.P_BEGIN and next_token.kind == TOK.P_END:
                    # P_BEGIN immediately followed by P_END:
                    # skip both and continue
                    token = None # Make sure we have correct status if next() raises StopIteration
                    token = next(token_stream)
                    continue
            else:
                if not in_sentence:
                    # This token starts a new sentence
                    yield tok_begin_sentence
                    in_sentence = True
                if token.kind == TOK.PUNCTUATION and token.txt in END_OF_SENTENCE:
                    # We may be finishing a sentence with not only a period but also
                    # right parenthesis and quotation marks
                    while next_token.kind == TOK.PUNCTUATION and next_token.txt in SENTENCE_FINISHERS:
                        yield token
                        token = next_token
                        next_token = next(token_stream)
                    # The sentence is definitely finished now
                    yield token
                    token = tok_end_sentence
                    in_sentence = False

            yield token
            token = next_token

    except StopIteration:
        pass

    # Final token (previous lookahead)
    if token is not None:
        if not in_sentence and token.kind != TOK.P_END and token.kind != TOK.S_END:
            # Starting something here
            yield tok_begin_sentence
            in_sentence = True
        yield token
        if in_sentence and (token.kind == TOK.P_END or token.kind == TOK.S_END):
            in_sentence = False

    # Done with the input stream
    # If still inside a sentence, finish it
    if in_sentence:
        yield tok_end_sentence


def parse_errors_1(token_stream):
    token = None
    try:
        # Maintain a one-token lookahead
        token = next(token_stream)
        while True:
            next_token = next(token_stream)
            # Make the lookahead checks we're interested in

            # Word reduplication; GrammCorr 1B
            if token == next_token and token.txt not in ALLOWED_MULTIPLES and token.kind == TOK.WORD:
                # coalesce and step to next token
                next_token = TOK.Word(next_token.txt, None, error=compound_error([[2], token.error, next_token.error]))
                token = next_token
                continue
            # Splitting wrongly compounded words; GrammCorr 1A
            if token.txt and token.txt.lower() in NOT_COMPOUNDS:
                for phrase_part in NOT_COMPOUNDS[token.txt.lower()]:
                    new_token = TOK.Word(phrase_part, None, error=4)
                    yield new_token
                token.error.append(4)
                token = next_token
                continue

            # Unite wrongly split compounds; GrammCorr 1X
            if (token.txt, next_token.txt) in SPLIT_COMPOUNDS:
                token = TOK.Word(token.txt + next_token.txt, None, error=compound_error([token.error, [5], next_token.error]))
                continue

            # Yield the current token and advance to the lookahead
            yield token
            token = next_token



    except StopIteration:
        # Final token (previous lookahead)
        if token:
            yield token


def test(token_stream): 
    token = None
    try:
        # Maintain a one-token lookahead
        token = next(token_stream)
        while True:
            yield token
            token = next(token_stream)
    except StopIteration:
        # Final token (previous lookahead)
        if token:
            yield token

def compound_error(toks):
    comp_err = []
    for alist in toks:
        if alist:
            comp_err.extend(alist)
    return comp_err

def annotate(token_stream, auto_uppercase):
    """ Look up word forms in the BIN word database. If auto_uppercase
        is True, change lower case words to uppercase if it looks likely
        that they should be uppercase. """
    at_sentence_start = False

    with BIN_Db.get_db() as db:

        # Consume the iterable source in wlist (which may be a generator)
        for t in token_stream:
            if t.kind != TOK.WORD:
                # Not a word: relay the token unchanged
                yield t
                if t.kind == TOK.S_BEGIN or (t.kind == TOK.PUNCTUATION and t.txt == ':'):
                    at_sentence_start = True
                elif t.kind != TOK.PUNCTUATION and t.kind != TOK.ORDINAL:
                    at_sentence_start = False
                continue
            if t.val is None:
                # Look up word in BIN database
                w, m = db.lookup_word(t.txt, at_sentence_start, auto_uppercase)
                # Yield a word tuple with meanings
                yield TOK.Word(w, m, error=[])
            else:
                # Already have a meaning
                yield t
            # No longer at sentence start
            at_sentence_start = False


# Recognize words that multiply numbers
MULTIPLIERS = {
    #"núll": 0,
    #"hálfur": 0.5,
    #"helmingur": 0.5,
    #"þriðjungur": 1.0 / 3,
    #"fjórðungur": 1.0 / 4,
    #"fimmtungur": 1.0 / 5,
    "einn": 1,
    "tveir": 2,
    "þrír": 3,
    "fjórir": 4,
    "fimm": 5,
    "sex": 6,
    "sjö": 7,
    "átta": 8,
    "níu": 9,
    "tíu": 10,
    "ellefu": 11,
    "tólf": 12,
    "þrettán": 13,
    "fjórtán": 14,
    "fimmtán": 15,
    "sextán": 16,
    "sautján": 17,
    "seytján": 17,
    "átján": 18,
    "nítján": 19,
    "tuttugu": 20,
    "þrjátíu": 30,
    "fjörutíu": 40,
    "fimmtíu": 50,
    "sextíu": 60,
    "sjötíu": 70,
    "áttatíu": 80,
    "níutíu": 90,
    #"par": 2,
    #"tugur": 10,
    #"tylft": 12,
    "hundrað": 100,
    "þúsund": 1000, # !!! Bæði hk og kvk!
    "þús.": 1000,
    "milljón": 1e6,
    "milla": 1e6,
    "milljarður": 1e9,
    "miljarður": 1e9,
    "ma.": 1e9
}

# Recognize words for fractions
FRACTIONS = {
    "þriðji": 1.0 / 3,
    "fjórði": 1.0 / 4,
    "fimmti": 1.0 / 5,
    "sjötti": 1.0 / 6,
    "sjöundi": 1.0 / 7,
    "áttundi": 1.0 / 8,
    "níundi": 1.0 / 9,
    "tíundi": 1.0 / 10,
    "tuttugasti": 1.0 / 20,
    "hundraðasti": 1.0 / 100,
    "þúsundasti": 1.0 / 1000,
    "milljónasti": 1.0 / 1e6
}

# Recognize words for percentages
PERCENTAGES = {
    "prósent": 1,
    "prósenta": 1,
    "hundraðshluti": 1,
    "prósentustig": 1
}

# Recognize words for nationalities (used for currencies)
NATIONALITIES = {
    "danskur": "dk",
    "enskur": "uk",
    "breskur": "uk",
    "bandarískur": "us",
    "kanadískur": "ca",
    "svissneskur": "ch",
    "sænskur": "se",
    "norskur": "no",
    "japanskur": "jp",
    "íslenskur": "is",
    "pólskur": "po",
    "kínverskur": "cn",
    "ástralskur": "au",
    "rússneskur": "ru",
    "indverskur": "in",
    "indónesískur": "id"
}

# Recognize words for currencies
CURRENCIES = {
    "króna": "ISK",
    "ISK": "ISK",
    "[kr.]": "ISK",
    "kr.": "ISK",
    "kr": "ISK",
    "pund": "GBP",
    "sterlingspund": "GBP",
    "GBP": "GBP",
    "dollari": "USD",
    "dalur": "USD",
    "bandaríkjadalur": "USD",
    "USD": "USD",
    "franki": "CHF",
    "rúbla": "RUB",
    "RUB": "RUB",
    "rúpía": "INR",
    "INR": "INR",
    "IDR": "IDR",
    "CHF": "CHF",
    "jen": "JPY",
    "yen": "JPY",
    "JPY": "JPY",
    "zloty": "PLN",
    "PLN": "PLN",
    "júan": "CNY",
    "yuan": "CNY",
    "CNY": "CNY",
    "evra": "EUR",
    "EUR": "EUR"
}

# Valid currency combinations
ISO_CURRENCIES = {
    ("dk", "ISK"): "DKK",
    ("is", "ISK"): "ISK",
    ("no", "ISK"): "NOK",
    ("se", "ISK"): "SEK",
    ("uk", "GBP"): "GBP",
    ("us", "USD"): "USD",
    ("ca", "USD"): "CAD",
    ("au", "USD"): "AUD",
    ("ch", "CHF"): "CHF",
    ("jp", "JPY"): "JPY",
    ("po", "PLN"): "PLN",
    ("ru", "RUB"): "RUB",
    ("in", "INR"): "INR", # Indian rupee
    ("id", "INR"): "IDR", # Indonesian rupiah
    ("cn", "CNY"): "CNY"
}

# Amount abbreviations including 'kr' for the ISK
# Corresponding abbreviations are found in Abbrev.conf
AMOUNT_ABBREV = {
    "þ.kr.": 1e3,
    "þús.kr.": 1e3,
    "m.kr.": 1e6,
    "mkr.": 1e6,
    "millj.kr.": 1e6,
    "mljó.kr.": 1e6,
    "ma.kr.": 1e9,
    "mö.kr.": 1e9,
    "mlja.kr.": 1e9
}

# Number words can be marked as subjects (any gender) or as numbers
NUMBER_CATEGORIES = frozenset(["töl", "to", "kk", "kvk", "hk", "lo"])


def match_stem_list(token, stems, filter_func=None):
    """ Find the stem of a word token in given dict, or return None if not found """
    if token.kind != TOK.WORD:
        return None
    # Go through the meanings with their stems
    if token.val:
        for m in token.val:
            # If a filter function is given, pass candidates to it
            try:
                lower_stofn = m.stofn.lower()
                if lower_stofn in stems and (filter_func is None or filter_func(m)):
                    return stems[lower_stofn]
            except Exception as e:
                print("Exception {0} in match_stem_list\nToken: {1}\nStems: {2}".format(e, token, stems))
                raise
    # No meanings found: this might be a foreign or unknown word
    # However, if it is still in the stems list we return True
    return stems.get(token.txt.lower(), None)


def case(bin_spec, default="nf"):
    """ Return the case specified in the bin_spec string """
    c = default
    if "NF" in bin_spec:
        c = "nf"
    elif "ÞF" in bin_spec:
        c = "þf"
    elif "ÞGF" in bin_spec:
        c = "þgf"
    elif "EF" in bin_spec:
        c = "ef"
    return c


def add_cases(cases, bin_spec, default="nf"):
    """ Add the case specified in the bin_spec string, if any, to the cases set """
    c = case(bin_spec, default)
    if c:
        cases.add(c)


def all_cases(token):
    """ Return a list of all cases that the token can be in """
    cases = set()
    if token.kind == TOK.WORD:
        # Roll through the potential meanings and extract the cases therefrom
        if token.val:
            for m in token.val:
                if m.fl == "ob":
                    # One of the meanings is an undeclined word: all cases apply
                    cases = ALL_CASES
                    break
                add_cases(cases, m.beyging, None)
    return list(cases)


def all_common_cases(token1, token2):
    """ Compute intersection of case sets for two tokens """
    set1 = set(all_cases(token1))
    set2 = set(all_cases(token2))
    return list(set1 & set2)


_GENDER_SET = { "kk", "kvk", "hk" }
_GENDER_DICT = { "KK": "kk", "KVK": "kvk", "HK": "hk" }

def all_genders(token):
    """ Return a list of the possible genders of the word in the token, if any """
    if token.kind != TOK.WORD:
        return None
    g = set()
    if token.val:
        for meaning in token.val:

            def find_gender(m):
                if m.ordfl in _GENDER_SET:
                    return m.ordfl # Plain noun
                # Probably number word ('töl' or 'to'): look at its spec
                for k, v in _GENDER_DICT.items():
                    if k in m.beyging:
                        return v
                return None

            gn = find_gender(meaning)
            if gn is not None:
               g.add(gn)
    return list(g)


def parse_phrases_1(token_stream):

    """ Parse a stream of tokens looking for phrases and making substitutions.
        First pass
    """
    with BIN_Db.get_db() as db:

        token = None
        try:

            # Maintain a one-token lookahead
            token = next(token_stream)
            while True:
                next_token = next(token_stream)
                # Logic for numbers and fractions that are partially or entirely
                # written out in words

                def number(tok):
                    """ If the token denotes a number, return that number - or None """
                    if tok.txt.lower() == "áttu":
                        # Do not accept 'áttu' (stem='átta', no kvk) as a number
                        return None
                    return match_stem_list(tok, MULTIPLIERS,
                        filter_func = lambda m: m.ordfl in NUMBER_CATEGORIES)

                def fraction(tok):
                    """ If the token denotes a fraction, return a corresponding number - or None """
                    return match_stem_list(tok, FRACTIONS)

                # Check whether we have an initial number word
                multiplier = number(token) if token.kind == TOK.WORD else None

                # Check for [number] 'hundred|thousand|million|billion'
                while (token.kind == TOK.NUMBER or multiplier is not None) \
                    and next_token.kind == TOK.WORD:

                    multiplier_next = number(next_token)

                    def convert_to_num(token):
                        if multiplier is not None:
                            token = TOK.Number(token.txt, multiplier,
                                all_cases(token), all_genders(token), error=token.error)
                        return token

                    if multiplier_next is not None:
                        # Retain the case of the last multiplier, except
                        # if it is possessive (eignarfall) and the previous
                        # token had a case ('hundruðum milljarða' is dative,
                        # not possessive)
                        next_case = all_cases(next_token)
                        next_gender = all_genders(next_token)
                        if "ef" in next_case:
                            # We may have something like 'hundruðum milljarða':
                            # use the case and gender of 'hundruðum', not 'milljarða'
                            next_case = all_cases(token) or next_case
                            next_gender = all_genders(token) or next_gender
                        token = convert_to_num(token)
                        token = TOK.Number(token.txt + " " + next_token.txt,
                            token.val[0] * multiplier_next,
                            next_case, next_gender, error=compound_error([token.error, next_token.error]))
                        # Eat the multiplier token
                        next_token = next(token_stream)
                    elif next_token.txt in AMOUNT_ABBREV:
                        # Abbreviations for ISK amounts
                        # For abbreviations, we do not know the case,
                        # but we try to retain the previous case information if any
                        token = convert_to_num(token)
                        token = TOK.Amount(token.txt + " " + next_token.txt, "ISK",
                            token.val[0] * AMOUNT_ABBREV[next_token.txt], # Number
                            token.val[1], token.val[2], error=compound_error([token.error, next_token.error])) # Cases and gender (and error type)
                        next_token = next(token_stream)
                    else:
                        # Check for [number] 'percent'
                        percentage = match_stem_list(next_token, PERCENTAGES)
                        if percentage is not None:
                            token = convert_to_num(token)
                            token = TOK.Percent(token.txt + " " + next_token.txt, token.val[0],
                                all_cases(next_token), all_genders(next_token), error=compound_error([token.error, next_token.error]))
                            # Eat the percentage token
                            next_token = next(token_stream)
                        else:
                            break

                    multiplier = None
                # DATEABS and DATEREL made
                # Check for [number | ordinal] [month name]
                if (token.kind == TOK.ORDINAL or token.kind == TOK.NUMBER or token.txt in DAYS_OF_MONTH) and next_token.kind == TOK.WORD:
                    month = match_stem_list(next_token, MONTHS)
                    if month is not None:
                        token = TOK.Date(token.txt + " " + next_token.txt, y = 0, m = month,
                            d = token.val if token.kind == TOK.ORDINAL else token.val[0] if token.kind == TOK.ORDINAL else DAYS_OF_MONTH[token.txt], error=compound_error([token.error, next_token.error]))
                        # Eat the month name token
                        next_token = next(token_stream)

                # Check for [DATE] [year]
                if token.kind == TOK.DATE and (next_token.kind == TOK.NUMBER or next_token.kind == TOK.YEAR):
                    if not token.val[0]:
                        # No year yet: add it
                        year = next_token.val if next_token.kind == TOK.YEAR else int(next_token.txt) if 1776 <= int(next_token.txt) <= 2100 else 0
                        token = TOK.Date(token.txt + " " + next_token.txt,
                            y = year, m = token.val[1], d = token.val[2], error=compound_error([token.error, next_token.error]))
                        # Eat the year token
                        next_token = next(token_stream)

                # Check for [month name] [year|YEAR]
                if token.kind == TOK.WORD and (next_token.kind == TOK.NUMBER or next_token.kind == TOK.YEAR):
                    month = match_stem_list(token, MONTHS)
                    if month is not None:
                        year = next_token.val if next_token.kind == TOK.YEAR else int(next_token.txt) if 1776 <= int(next_token.txt) <= 2100 else 0
                        token = TOK.Date(token.txt + " " + next_token.txt, y = year, m = month, d = 0, error=compound_error([token.error, next_token.error]))
                        # Eat the year token
                        next_token = next(token_stream)

                # Check for a single YEAR, change to DATEREL -- changed to keep distinction
                #if token.kind == TOK.YEAR:
                #    token = TOK.Daterel(token.txt, y = token.val, m = 0, d = 0)

                # Check for a single month, change to DATEREL
                if token.kind == TOK.WORD:
                    month = match_stem_list(token, MONTHS)
                    if month is not None:
                        token = TOK.Daterel(token.txt, y = 0, m = month, d = 0, error=token.error)

                # Split DATE into DATEABS and DATEREL
                if token.kind == TOK.DATE:
                    if token.val[0] and token.val[1] and token.val[2]:
                        token = TOK.Dateabs(token.txt, y = token.val[0], m = token.val[1], d = token.val[2], error=token.error)
                    else:
                        token = TOK.Daterel(token.txt, y = token.val[0], m = token.val[1], d = token.val[2], error=token.error)

                # Check for [date] [time] (absolute)
                if token.kind == TOK.DATEABS and next_token.kind == TOK.TIME:
                    # Create an absolute time stamp
                    y, mo, d = token.val
                    h, m, s = next_token.val
                    token = TOK.Timestampabs(token.txt + " " + next_token.txt,
                        y = y, mo = mo, d = d, h = h, m = m, s = s, error=compound_error([token.error, next_token.error]))
                    # Eat the time token
                    next_token = next(token_stream)
                # Check for [date] [time] (relative)
                if token.kind == TOK.DATEREL and next_token.kind == TOK.TIME:
                    # Create a time stamp
                    y, mo, d = token.val
                    h, m, s = next_token.val
                    token = TOK.Timestamprel(token.txt + " " + next_token.txt,
                        y = y, mo = mo, d = d, h = h, m = m, s = s, error=compound_error([token.error, next_token.error]))
                    # Eat the time token
                    next_token = next(token_stream)
                
                # Check for currency name doublets, for example
                # 'danish krona' or 'british pound'
                if token.kind == TOK.WORD and next_token.kind == TOK.WORD:
                    nat = match_stem_list(token, NATIONALITIES)
                    if nat is not None:
                        cur = match_stem_list(next_token, CURRENCIES)
                        if cur is not None:
                            if (nat, cur) in ISO_CURRENCIES:
                                # Match: accumulate the possible cases
                                token = TOK.Currency(token.txt + " "  + next_token.txt,
                                    ISO_CURRENCIES[(nat, cur)],
                                    all_common_cases(token, next_token),
                                    all_genders(next_token), error=compound_error([token.error, next_token.error]))
                                next_token = next(token_stream)

                # Check for composites:
                # 'stjórnskipunar- og eftirlitsnefnd'
                # 'viðskipta- og iðnaðarráðherra'
                # 'marg-ítrekaðri'
                if token.kind == TOK.WORD and \
                    next_token.kind == TOK.PUNCTUATION and next_token.txt == COMPOSITE_HYPHEN:

                    og_token = next(token_stream)
                    if og_token.kind != TOK.WORD or (og_token.txt != "og" and og_token.txt != "eða"):
                        # Incorrect prediction: make amends and continue
                        handled = False
                        if og_token.kind == TOK.WORD:
                            composite = token.txt + "-" + og_token.txt
                            if token.txt.lower() in ADJECTIVE_PREFIXES:
                                # hálf-opinberri, marg-ítrekaðri
                                token = TOK.Word(composite,
                                    [m for m in og_token.val if m.ordfl == "lo" or m.ordfl == "ao"], error=compound_error([token.error, next_token.error]))
                                next_token = next(token_stream)
                                handled = True
                            else:
                                # Check for Vestur-Þýskaland, Suður-Múlasýsla (which are in BÍN in their entirety)
                                m = db.meanings(composite)
                                if m:
                                    # Found composite in BÍN: return it as a single token
                                    token = TOK.Word(composite, m, error=compound_error([token.error, next_token.error]))
                                    next_token = next(token_stream)
                                    handled = True
                        if not handled:
                            yield token
                            # Put a normal hyphen instead of the composite one
                            token = TOK.Punctuation(HYPHEN, error=token.error)
                            next_token = og_token
                    else:
                        # We have 'viðskipta- og'
                        final_token = next(token_stream)
                        if final_token.kind != TOK.WORD:
                            # Incorrect: unwind
                            yield token
                            yield TOK.Punctuation(HYPHEN, error=[]) # Normal hyphen
                            token = og_token
                            next_token = final_token
                        else:
                            # We have 'viðskipta- og iðnaðarráðherra'
                            # Return a single token with the meanings of
                            # the last word, but an amalgamated token text.
                            # Note: there is no meaning check for the first
                            # part of the composition, so it can be an unknown word.
                            txt = token.txt + "- " + og_token.txt + \
                                " " + final_token.txt
                            token = TOK.Word(txt, final_token.val, error=compound_error([token.error, next_token.error, final_token.error]))
                            next_token = next(token_stream)

                # Yield the current token and advance to the lookahead
                yield token
                token = next_token

        except StopIteration:
            pass

        # Final token (previous lookahead)
        if token:
            yield token


def parse_phrases_2(token_stream):

    """ Parse a stream of tokens looking for phrases and making substitutions.
        Second pass
    """
    token = None
    try:

        # Maintain a one-token lookahead
        token = next(token_stream)
        # Maintain a set of full person names encountered
        names = set()

        at_sentence_start = False

        while True:
            next_token = next(token_stream)
            # Make the lookahead checks we're interested in

            # Check for [number] [currency] and convert to [amount]
            if token.kind == TOK.NUMBER and (next_token.kind == TOK.WORD or
                next_token.kind == TOK.CURRENCY):

                # Preserve the case of the number, if available
                # (milljónir, milljóna, milljónum)
                cases = token.val[1]
                genders = token.val[2]
                cur = None

                if next_token.kind == TOK.WORD:
                    # Try to find a currency name
                    cur = match_stem_list(next_token, CURRENCIES)
                    if cur is None and next_token.txt.isupper():
                        # Might be an ISO abbrev (which is not in BÍN)
                        cur = CURRENCIES.get(next_token.txt)
                        if not cases:
                            cases = list(ALL_CASES)
                        if not genders:
                            genders = ["hk"]
                    if cur is not None:
                        # Use the case and gender information from the currency name
                        if not cases:
                            cases = all_cases(next_token)
                        if not genders:
                            genders = all_genders(next_token)
                elif next_token.kind == TOK.CURRENCY:
                    # Already have an ISO identifier for a currency
                    cur = next_token.val[0]
                    if next_token.val[1]:
                        cases = next_token.val[1]
                    if next_token.val[2]:
                        genders = next_token.val[2]

                if cur is not None:
                    # Create an amount
                    # Use the case and gender information from the number, if any
                    token = TOK.Amount(token.txt + " " + next_token.txt,
                        cur, token.val[0], cases, genders, error=compound_error([token.error, next_token.error]))
                    # Eat the currency token
                    next_token = next(token_stream)

            # Check for [time] [date] (absolute)
            if token.kind == TOK.TIME and next_token.kind == TOK.DATEABS:
                # Create a time stamp
                h, m, s = token.val
                y, mo, d = next_token.val
                token = TOK.Timestampabs(token.txt + " " + next_token.txt,
                    y = y, mo = mo, d = d, h = h, m = m, s = s, error=compound_error([token.error, next_token.error]))
                # Eat the time token
                next_token = next(token_stream)

            # Check for [time] [date] (relative)
            if token.kind == TOK.TIME and next_token.kind == TOK.DATEREL:
                # Create a time stamp
                h, m, s = token.val
                y, mo, d = next_token.val
                token = TOK.Timestamprel(token.txt + " " + next_token.txt,
                    y = y, mo = mo, d = d, h = h, m = m, s = s, error=compound_error([token.error, next_token.error]))
                # Eat the time token
                next_token = next(token_stream)

            # Logic for human names

            def stems(tok, categories, given_name = False):
                """ If the token denotes a given name, return its possible
                    interpretations, as a list of PersonName tuples (name, case, gender).
                    If first_name is True, we omit from the list all name forms that
                    occur in the disallowed_names section in the configuration file. """
                if tok.kind != TOK.WORD or not tok.val:
                    return None
                if at_sentence_start and tok.txt in NOT_NAME_AT_SENTENCE_START:
                    # Disallow certain person names at the start of sentences,
                    # such as 'Annar'
                    return None
                # Set up the names we're not going to allow
                dstems = DisallowedNames.STEMS if given_name else { }
                # Look through the token meanings
                result = []
                for m in tok.val:
                    if m.fl in categories and "ET" in m.beyging:
                        # If this is a given name, we cut out name forms
                        # that are frequently ambiguous and wrong, i.e. "Frá" as accusative
                        # of the name "Frár", and "Sigurð" in the nominative.
                        c = case(m.beyging)
                        if m.stofn not in dstems or c not in dstems[m.stofn]:
                            # Note the stem ('stofn') and the gender from the word type ('ordfl')
                            result.append(PersonName(name = m.stofn, gender = m.ordfl, case = c))
                return result if result else None

            def has_category(tok, categories):
                """ Return True if the token matches a meaning with any of the given categories """
                if tok.kind != TOK.WORD or not tok.val:
                    return False
                return any(m.fl in categories for m in tok.val)

            def has_other_meaning(tok, category):
                """ Return True if the token can denote something besides a given name """
                if tok.kind != TOK.WORD or not tok.val:
                    return True
                # Return True if there is a different meaning, not a given name
                return any(m.fl != category for m in tok.val)

            # Check for person names
            def given_names(tok):
                """ Check for Icelandic person name (category 'ism') """
                if tok.kind != TOK.WORD or not tok.txt[0].isupper():
                    # Must be a word starting with an uppercase character
                    return None
                return stems(tok, {"ism"}, given_name = True)

            # Check for surnames
            def surnames(tok):
                """ Check for Icelandic patronym (category 'föð') or matronym (category 'móð') """
                if tok.kind != TOK.WORD or not tok.txt[0].isupper():
                    # Must be a word starting with an uppercase character
                    return None
                return stems(tok, {"föð", "móð"})

            # Check for unknown surnames
            def unknown_surname(tok):
                """ Check for unknown (non-Icelandic) surnames """
                # Accept (most) upper case words as a surnames
                if tok.kind != TOK.WORD:
                    return False
                if not tok.txt[0].isupper():
                    # Must start with capital letter
                    return False
                if has_category(tok, {"föð", "móð"}):
                    # This is a known surname, not an unknown one
                    return False
                # Allow single-letter abbreviations, but not multi-letter
                # all-caps words (those are probably acronyms)
                return len(tok.txt) == 1 or not tok.txt.isupper()

            def given_names_or_middle_abbrev(tok):
                """ Check for given name or middle abbreviation """
                gnames = given_names(tok)
                if gnames is not None:
                    return gnames
                if tok.kind != TOK.WORD:
                    return None
                wrd = tok.txt
                if wrd.startswith('['):
                    # Abbreviation: Cut off the brackets & trailing period, if present
                    if wrd.endswith('.]'):
                        wrd = wrd[1:-2]
                    else:
                        # This is probably a C. which had its period cut off as a sentence ending...
                        wrd = wrd[1:-1]
                if len(wrd) > 2 or not wrd[0].isupper():
                    if wrd not in { "van", "de", "den", "der", "el", "al" }: # "of" was here
                        # Accept "Thomas de Broglie", "Ruud van Nistelroy"
                        return None
                # One or two letters, capitalized: accept as middle name abbrev,
                # all genders and cases possible
                return [ PersonName(name = wrd, gender = None, case = None) ]

            def compatible(pn, npn):
                """ Return True if the next PersonName (np) is compatible with the one we have (p) """
                if npn.gender and (npn.gender != pn.gender):
                    return False
                if npn.case and (npn.case != pn.case):
                    return False
                return True

            if token.kind == TOK.WORD and token.val and token.val[0].fl == "nafn":
                # Convert a WORD with fl="nafn" to a PERSON with the correct gender, in all cases
                gender = token.val[0].ordfl
                token = TOK.Person(token.txt, [ PersonName(token.txt, gender, case) for case in ALL_CASES ], error=token.error)
                gn = None
            else:
                gn = given_names(token)

            if gn:
                # Found at least one given name: look for a sequence of given names
                # having compatible genders and cases
                w = token.txt
                patronym = False
                while True:
                    ngn = given_names_or_middle_abbrev(next_token)
                    if not ngn:
                        break
                    # Look through the stuff we got and see what is compatible
                    r = []
                    for p in gn:
                        # noinspection PyTypeChecker
                        for np in ngn:
                            if compatible(p, np):
                                # Compatible: add to result
                                r.append(PersonName(name = p.name + " " + np.name, gender = p.gender, case = p.case))
                    if not r:
                        # This next name is not compatible with what we already
                        # have: break
                        break
                    # Success: switch to new given name list
                    gn = r
                    w += " " + (ngn[0].name if next_token.txt[0] == '[' else next_token.txt)
                    next_token = next(token_stream)

                # Check whether the sequence of given names is followed
                # by one or more surnames (patronym/matronym) of the same gender,
                # for instance 'Dagur Bergþóruson Eggertsson'

                def eat_surnames(gn, w, patronym, next_token):
                    """ Process contiguous known surnames, typically "*dóttir/*son", while they are
                        compatible with the given name we already have """
                    while True:
                        sn = surnames(next_token)
                        if not sn:
                            break
                        r = []
                        # Found surname: append it to the accumulated name, if compatible
                        for p in gn:
                            for np in sn:
                                if compatible(p, np):
                                    r.append(PersonName(name = p.name + " " + np.name, gender = np.gender, case = np.case))
                        if not r:
                            break
                        # Compatible: include it and advance to the next token
                        gn = r
                        w += " " + next_token.txt
                        patronym = True
                        next_token = next(token_stream)
                    return gn, w, patronym, next_token

                gn, w, patronym, next_token = eat_surnames(gn, w, patronym, next_token)

                # Must have at least one possible name
                assert len(gn) >= 1

                if not patronym:
                    # We stop name parsing after we find one or more Icelandic
                    # patronyms/matronyms. Otherwise, check whether we have an
                    # unknown uppercase word next;
                    # if so, add it to the person names we've already found
                    while unknown_surname(next_token):
                        for ix, p in enumerate(gn):
                            gn[ix] = PersonName(name = p.name + " " + next_token.txt, gender = p.gender, case = p.case)
                        w += " " + next_token.txt
                        next_token = next(token_stream)
                        # Assume we now have a patronym
                        patronym = True

                    if patronym:
                        # We still might have surnames coming up: eat them too, if present
                        gn, w, _, next_token = eat_surnames(gn, w, patronym, next_token)

                found_name = False
                # If we have a full name with patronym, store it
                if patronym:
                    names |= set(gn)
                else:
                    # Look through earlier full names and see whether this one matches
                    for ix, p in enumerate(gn):
                        gnames = p.name.split(' ') # Given names
                        for lp in names:
                            match = (not p.gender) or (p.gender == lp.gender)
                            if match:
                                # The gender matches
                                lnames = set(lp.name.split(' ')[0:-1]) # Leave the patronym off
                                for n in gnames:
                                    if n not in lnames:
                                        # We have a given name that does not match the person
                                        match = False
                                        break
                            if match:
                                # All given names match: assign the previously seen full name
                                gn[ix] = PersonName(name = lp.name, gender = lp.gender, case = p.case)
                                found_name = True
                                break

                # If this is not a "strong" name, backtrack from recognizing it.
                # A "weak" name is (1) at the start of a sentence; (2) only one
                # word; (3) that word has a meaning that is not a name;
                # (4) the name has not been seen in a full form before;
                # (5) not on a 'well known name' list.

                weak = at_sentence_start and (' ' not in w) and not patronym and \
                    not found_name and (has_other_meaning(token, "ism") and w not in NamePreferences.SET)

                if not weak:
                    # Return a person token with the accumulated name
                    # and the intersected set of possible cases
                    token = TOK.Person(w, gn, error=token.error)

            # Yield the current token and advance to the lookahead
            yield token

            if token.kind == TOK.S_BEGIN or (token.kind == TOK.PUNCTUATION and token.txt == ':'):
                at_sentence_start = True
            elif token.kind != TOK.PUNCTUATION and token.kind != TOK.ORDINAL:
                at_sentence_start = False
            token = next_token

    except StopIteration:
        pass

    # Final token (previous lookahead)
    if token:
        yield token


def parse_static_phrases(token_stream, auto_uppercase):

    """ Parse a stream of tokens looking for static multiword phrases
        (i.e. phrases that are not affected by inflection).
        The algorithm implements N-token lookahead where N is the
        length of the longest phrase.
    """
    tq = [] # Token queue
    state = defaultdict(list) # Phrases we're considering
    pdict = StaticPhrases.DICT # The phrase dictionary
    try:

        while True:

            token = next(token_stream)
            if token.txt is None: # token.kind != TOK.WORD:
                # Not a word: no match; discard state
                if tq:
                    for t in tq: yield t
                    tq = []
                if state:
                    state = defaultdict(list)
                yield token
                continue

            # Look for matches in the current state and build a new state
            newstate = defaultdict(list)
            wo = token.txt # Original word
            w = wo.lower() # Lower case
            if wo == w:
                wo = w

            def add_to_state(slist, index):
                """ Add the list of subsequent words to the new parser state """
                wrd = slist[0]
                rest = slist[1:]
                newstate[wrd].append((rest, index))

            # First check for original (uppercase) word in the state, if any;
            # if that doesn't match, check the lower case
            wm = None
            if wo is not w and wo in state:
                wm = wo
            elif w in state:
                wm = w

            if wm:
                # This matches an expected token:
                # go through potential continuations
                tq.append(token) # Add to lookahead token queue
                token = None
                for sl, ix in state[wm]:
                    if not sl:
                        # No subsequent word: this is a complete match
                        # Reconstruct original text behind phrase
                        plen = StaticPhrases.get_length(ix)
                        while len(tq) > plen:
                            # We have extra queued tokens in the token queue
                            # that belong to a previously seen partial phrase
                            # that was not completed: yield them first
                            yield tq.pop(0)
                        w = " ".join([ t.txt for t in tq ])
                        werr = [ t.error for t in tq ]
                        # Add the entire phrase as one 'word' to the token queue
                        yield TOK.Word(w,
                            [ BIN_Meaning._make(r)
                                for r in StaticPhrases.get_meaning(ix) ], error=werr)
                        # Discard the state and start afresh
                        newstate = defaultdict(list)
                        w = wo = ""
                        tq = []
                        werr = []
                        # Note that it is possible to match even longer phrases
                        # by including a starting phrase in its entirety in
                        # the static phrase dictionary
                        break
                    add_to_state(sl, ix)
            elif tq:
                for t in tq: yield t
                tq = []

            wm = None
            if auto_uppercase and len(wo) == 1 and w is wo:
                # If we are auto-uppercasing, leave single-letter lowercase
                # phrases alone, i.e. 'g' for 'gram' and 'm' for 'meter'
                pass
            elif wo is not w and wo in pdict:
                wm = wo
            elif w in pdict:
                wm = w

            # Add all possible new states for phrases that could be starting
            if wm:
                # This word potentially starts a phrase
                for sl, ix in pdict[wm]:
                    if not sl:
                        # Simple replace of a single word
                        if tq:
                            for t in tq: yield tq
                            tq = []
                        # Yield the replacement token
                        yield TOK.Word(token.txt,
                            [ BIN_Meaning._make(r)
                                for r in StaticPhrases.get_meaning(ix) ], error=[])
                        newstate = defaultdict(list)
                        token = None
                        break
                    add_to_state(sl, ix)
                if token:
                    tq.append(token)
            elif token:
                yield token

            # Transition to the new state
            state = newstate

    except StopIteration:
        # Token stream is exhausted
        pass

    # Yield any tokens remaining in queue
    for t in tq: yield t


def disambiguate_phrases(token_stream):

    """ Parse a stream of tokens looking for common ambiguous multiword phrases
        (i.e. phrases that have a well known very likely interpretation but
        other extremely uncommon ones are also grammatically correct).
        The algorithm implements N-token lookahead where N is the
        length of the longest phrase.
    """
    tq = [] # Token queue
    state = defaultdict(list) # Phrases we're considering
    pdict = AmbigPhrases.DICT # The phrase dictionary

    try:

        while True:

            token = next(token_stream)

            if token.kind != TOK.WORD:
                # Not a word: no match; yield the token queue
                if tq:
                    for t in tq: yield t
                    tq = []
                # Discard the previous state, if any
                if state:
                    state = defaultdict(list)
                # ...and yield the non-matching token
                yield token
                continue

            # Look for matches in the current state and build a new state
            newstate = defaultdict(list)
            w = token.txt.lower()

            def add_to_state(slist, index):
                """ Add the list of subsequent words to the new parser state """
                wrd = slist[0]
                rest = slist[1:]
                newstate[wrd].append((rest, index))

            if w in state:
                # This matches an expected token:
                # go through potential continuations
                tq.append(token) # Add to lookahead token queue
                token = None
                for sl, ix in state[w]:
                    if not sl:
                        # No subsequent word: this is a complete match
                        # Discard meanings of words in the token queue that are not
                        # compatible with the category list specified
                        cats = AmbigPhrases.get_cats(ix)
                        for t, cat in zip(tq, cats):
                            # Yield a new token with fewer meanings for each original token in the queue
                            if cat == "fs":
                                # Handle prepositions specially, since we may have additional
                                # preps defined in Main.conf that don't have fs meanings in BÍN
                                w = t.txt.lower()
                                yield TOK.Word(t.txt, [ BIN_Meaning(w, 0, "fs", "alm", w, "-") ], error=[])
                            else:
                                yield TOK.Word(t.txt, [m for m in t.val if m.ordfl == cat], error=[])

                        # Discard the state and start afresh
                        if newstate:
                            newstate = defaultdict(list)
                        w = ""
                        tq = []
                        # Note that it is possible to match even longer phrases
                        # by including a starting phrase in its entirety in
                        # the static phrase dictionary
                        break
                    add_to_state(sl, ix)
            elif tq:
                # This does not continue a started phrase:
                # yield the accumulated token queue
                for t in tq: yield t
                tq = []

            if w in pdict:
                # This word potentially starts a new phrase
                for sl, ix in pdict[w]:
                    # assert sl
                    add_to_state(sl, ix)
                if token:
                    tq.append(token) # Start a lookahead queue with this token
            elif token:
                # Not starting a new phrase: pass the token through
                yield token

            # Transition to the new state
            state = newstate

    except StopIteration:
        # Token stream is exhausted
        pass
    # Yield any tokens remaining in queue
    for t in tq: yield t


def recognize_entities(token_stream, enclosing_session = None):

    """ Parse a stream of tokens looking for (capitalized) entity names
        The algorithm implements N-token lookahead where N is the
        length of the longest entity name having a particular initial word.
    """
    tq = [] # Token queue
    state = defaultdict(list) # Phrases we're considering
    ecache = dict() # Entitiy definition cache
    lastnames = dict() # Last name to full name mapping ('Clinton' -> 'Hillary Clinton')

    with BIN_Db.get_db() as db, \
        SessionContext(session = enclosing_session, commit = True, read_only = True) as session:

        def fetch_entities(w, fuzzy = True):
            """ Return a list of entities matching the word(s) given,
                exactly if fuzzy = False, otherwise also as a starting word(s) """
            q = session.query(Entity.name, Entity.verb, Entity.definition)
            if fuzzy:
                q = q.filter(Entity.name.like(w + " %") | (Entity.name == w))
            else:
                q = q.filter(Entity.name == w)
            return q.all()

        def query_entities(w):
            """ Return a list of entities matching the initial word given """
            e = ecache.get(w)
            if e is None:
                ecache[w] = e = fetch_entities(w)
            return e

        def lookup_lastname(lastname):
            """ Look up a last name in the lastnames registry,
                eventually without a possessive 's' at the end, if present """
            fullname = lastnames.get(lastname)
            if fullname is not None:
                # Found it
                return fullname
            # Try without a possessive 's', if present
            if len(lastname) > 1 and lastname[-1] == 's':
                return lastnames.get(lastname[0:-1])
            # Nope, no match
            return None

        def flush_match():
            """ Flush a match that has been accumulated in the token queue """
            if len(tq) == 1 and lookup_lastname(tq[0].txt) is not None:
                # If single token, it may be the last name of a
                # previously seen entity or person
                return token_or_entity(tq[0])
            # Reconstruct original text behind phrase
            ename = " ".join([t.txt for t in tq])
            origerror = [t.error for t in tq]
            # We don't include the definitions in the token - they should be looked up
            # on the fly when processing or displaying the parsed article
            return TOK.Entity(ename, None, error=origerror)

        def token_or_entity(token):
            """ Return a token as-is or, if it is a last name of a person that has already
                been mentioned in the token stream by full name, refer to the full name """
            assert token.txt[0].isupper()
            tfull = lookup_lastname(token.txt)
            if tfull is None:
                # Not a last name of a previously seen full name
                return token
            if tfull.kind != TOK.PERSON:
                # Return an entity token with no definitions
                # (this will eventually need to be looked up by full name when
                # displaying or processing the article)
                return TOK.Entity(token.txt, None, error=token.error)
            # Return the full name meanings
            return TOK.Person(token.txt, tfull.val, error=token.error)

        try:

            while True:

                token = next(token_stream)

                if not token.txt: # token.kind != TOK.WORD:
                    if state:
                        if None in state:
                            yield flush_match()
                        else:
                            for t in tq:
                                yield t
                        tq = []
                        state = defaultdict(list)
                    yield token
                    continue

                # Look for matches in the current state and build a new state
                newstate = defaultdict(list)
                w = token.txt # Original word

                def add_to_state(slist, entity):
                    """ Add the list of subsequent words to the new parser state """
                    wrd = slist[0] if slist else None
                    rest = slist[1:]
                    newstate[wrd].append((rest, entity))

                if w in state:
                    # This matches an expected token
                    tq.append(token) # Add to lookahead token queue
                    # Add the matching tails to the new state
                    for sl, entity in state[w]:
                        add_to_state(sl, entity)
                    # Update the lastnames mapping
                    fullname = " ".join([t.txt for t in tq])
                    parts = fullname.split()
                    # If we now have 'Hillary Rodham Clinton',
                    # make sure we delete the previous 'Rodham' entry
                    for p in parts[1:-1]:
                        if p in lastnames:
                            del lastnames[p]
                    if parts[-1][0].isupper():
                        # 'Clinton' -> 'Hillary Rodham Clinton'
                        lastnames[parts[-1]] = TOK.Entity(fullname, None, error=token.error)
                else:
                    # Not a match for an expected token
                    if state:
                        if None in state:
                            # Flush the already accumulated match
                            yield flush_match()
                        else:
                            for t in tq:
                                yield t
                        tq = []

                    # Add all possible new states for entity names that could be starting
                    weak = True
                    cnt = 1
                    upper = w and w[0].isupper()
                    parts = None

                    if upper and " " in w:
                        # For all uppercase phrases (words, entities, persons),
                        # maintain a map of last names to full names
                        parts = w.split()
                        lastname = parts[-1]
                        # Clinton -> Hillary [Rodham] Clinton
                        if lastname[0].isupper():
                            # Look for Icelandic patronyms/matronyms
                            _, m = db.lookup_word(lastname, False)
                            if m and any(mm.fl in { "föð", "móð" } for mm in m):
                                # We don't store Icelandic patronyms/matronyms as surnames
                                pass
                            else:
                                lastnames[lastname] = token

                    if token.kind == TOK.WORD and upper and w not in Abbreviations.DICT:
                        if " " in w:
                            # w may be a person name with more than one embedded word
                            # parts is assigned in the if statement above
                            cnt = len(parts)
                        elif not token.val or ('-' in token.val[0].stofn):
                            # No BÍN meaning for this token, or the meanings were constructed
                            # by concatenation (indicated by a hyphen in the stem)
                            weak = False # Accept single-word entity references
                        # elist is a list of Entity instances
                        elist = query_entities(w)
                    else:
                        elist = []

                    if elist:
                        # This word might be a candidate to start an entity reference
                        candidate = False
                        for e in elist:
                            sl = e.name.split()[cnt:] # List of subsequent words in entity name
                            if sl:
                                # Here's a candidate for a longer entity reference than we already have
                                candidate = True
                            if sl or not weak:
                                add_to_state(sl, e)
                        if weak and not candidate:
                            # Found no potential entity reference longer than this token
                            # already is - and we have a BÍN meaning for it: Abandon the effort
                            assert not newstate
                            assert not tq
                            yield token_or_entity(token)
                        else:
                            # Go for it: Initialize the token queue
                            tq = [ token ]
                    else:
                        # Not a start of an entity reference: simply yield the token
                        assert not tq
                        if upper:
                            # Might be a last name referring to a full name
                            yield token_or_entity(token)
                        else:
                            yield token

                # Transition to the new state
                state = newstate

        except StopIteration:
            # Token stream is exhausted
            pass

        # Yield an accumulated match if present
        if state:
            if None in state:
                yield flush_match()
            else:
                for t in tq:
                    yield t
            tq = []

    # print("\nEntity cache:\n{0}".format("\n".join("'{0}': {1}".format(k, v) for k, v in ecache.items())))
    # print("\nLast names:\n{0}".format("\n".join("{0}: {1}".format(k, v) for k, v in lastnames.items())))

    assert not tq


def raw_tokenize(text):
    """ Tokenize text up to but not including the BÍN annotation pass """
    token_stream = parse_tokens(text)
    token_stream = parse_particles(token_stream)
    token_stream = parse_sentences(token_stream)
    token_stream = parse_errors_1(token_stream)
    token_stream = test(token_stream)
    return token_stream


def tokenize(text, auto_uppercase = False, enclosing_session = None):
    """ Tokenize text in several phases, returning a generator (iterable sequence) of tokens
        that processes tokens on-demand. If auto_uppercase is True, the tokenizer
        attempts to correct lowercase words that probably should be uppercase. """

    # Thank you Python for enabling this programming pattern ;-)
    token_stream = raw_tokenize(text)

    # Static multiword phrases
    token_stream = parse_static_phrases(token_stream, auto_uppercase)

    # Lookup meanings from dictionary
    token_stream = annotate(token_stream, auto_uppercase)

    # First phrase pass
    token_stream = parse_phrases_1(token_stream)

    # Second phrase pass
    token_stream = parse_phrases_2(token_stream)

    # Recognize named entities from database
    token_stream = recognize_entities(token_stream, enclosing_session)

     # Eliminate very uncommon meanings
    token_stream = disambiguate_phrases(token_stream)

    token_stream = test_2(token_stream)


    return token_stream


def paragraphs(toklist):
    """ Generator yielding paragraphs from a token list. Each paragraph is a list
        of sentence tuples. Sentence tuples consist of the index of the first token
        of the sentence (the TOK.S_BEGIN token) and a list of the tokens within the
        sentence, not including the starting TOK.S_BEGIN or the terminating TOK.S_END
        tokens. """

    def valid_sent(sent):
        """ Return True if the token list in sent is a proper
            sentence that we want to process further """
        if not sent:
            return False
        # A sentence with only punctuation is not valid
        return any(t[0] != TOK.PUNCTUATION for t in sent)

    if not toklist:
        return
    sent = [] # Current sentence
    sent_begin = 0
    current_p = [] # Current paragraph

    for ix, t in enumerate(toklist):
        t0 = t[0]
        if t0 == TOK.S_BEGIN:
            sent = []
            sent_begin = ix
        elif t0 == TOK.S_END:
            if valid_sent(sent):
                # Do not include or count zero-length sentences
                current_p.append((sent_begin, sent))
            sent = []
        elif t0 == TOK.P_BEGIN or t0 == TOK.P_END:
            # New paragraph marker: Start a new paragraph if we didn't have one before
            # or if we already had one with some content
            if valid_sent(sent):
                current_p.append((sent_begin, sent))
            sent = []
            if current_p:
                yield current_p
                current_p = []
        else:
            sent.append(t)
    if valid_sent(sent):
        current_p.append((sent_begin, sent))
    if current_p:
        yield current_p


def canonicalize_token(t):
    """ Convert a token in-situ from a compact dictionary representation
        (typically created by TreeUtility._describe_token()) to a normalized,
        verbose form that is appropriate for external consumption """

    # Set the token kind to a readable string
    kind = t.get("k", TOK.WORD)
    t["k"] = TOK.descr[kind]
    terminal = None
    if "t" in t:
        terminal = t["t"]
        # Change "literal:category" to category,
        # or 'stem'_var1_var2 to category_var1_var2
        if terminal[0] in "\"'":
            # Convert 'literal'_var1_var2 to cat_var1_var2
            # Note that the literal can contain underscores!
            endq = terminal.rindex(terminal[0])
            first = terminal[0:endq + 1]
            rest = terminal[endq + 1:]
            if ':' in first:
                # The word category was given in the literal: use it
                # (In almost all cases this matches the meaning, but
                # 'stt' is an exception)
                cat_override = first.split(':')[-1][:-1]
                first = cat_override
            elif "m" in t:
                # Get the word category from the meaning
                first = t["m"][1]
            if first in { "kk", "kvk", "hk" }:
                first = "no_" + first
            t["t"] = first + rest
    if "m" in t:
        # Flatten the meaning from a tuple/list
        m = t["m"]
        del t["m"]
        # s = stofn (stem)
        # c = ordfl (category)
        # f = fl (class)
        # b = beyging (declination)
        t.update(dict(s = m[0], c = m[1], f = m[2], b = m[3]))
    if "v" in t:
        # Flatten and simplify the val field, if present
        val = t["v"]
        if kind == TOK.AMOUNT:
            # Flatten and simplify amounts
            t["v"] = dict(amount = val[0], currency = val[1])
        elif kind in { TOK.NUMBER, TOK.CURRENCY, TOK.PERCENT }:
            # Number, ISO currency code, percentage
            t["v"] = val[0]
        elif kind == TOK.DATE:
            t["v"] = dict(y = val[0], mo = val[1], d = val[2])
        elif kind == TOK.DATEABS:
            t["v"] = dict(y = val[0], mo = val[1], d = val[2])
        elif kind == TOK.DATEREL:
            t["v"] = dict(y = val[0], mo = val[1], d = val[2])
        elif kind == TOK.TIME:
            t["v"] = dict(h = val[0], m = val[1], s = val[2])
        elif kind == TOK.TIMESTAMP:
            t["v"] = dict(y = val[0], mo = val[1], d = val[2],
                h = val[3], m = val[4], s = val[5])
        elif kind == TOK.TIMESTAMPABS:
            t["v"] = dict(y = val[0], mo = val[1], d = val[2],
                h = val[3], m = val[4], s = val[5])
        elif kind == TOK.TIMESTAMPREL:
            t["v"] = dict(y = val[0], mo = val[1], d = val[2],
                h = val[3], m = val[4], s = val[5])
        elif kind == TOK.PERSON:
            # Move the nominal form of the name to the "s" (stem) field
            t["s"] = t["v"]
            del t["v"]
            # Move the gender to the "c" (category) field
            if "g" in t:
                t["c"] = t["g"]
                del t["g"]
    if (kind == TOK.ENTITY) or (kind == TOK.WORD) and "s" not in t:
        # Put in a stem for entities and proper names
        t["s"] = t["x"]


def stems_of_token(t):
    """ Return a list of word stem descriptors associated with the token t.
        This is an empty list if the token is not a word or person or entity name.
        The list can contain multiple stems, for instance in the case
        of composite words ('sjómannadagur' -> ['sjómannadagur/kk', sjómaður/kk', 'dagur/kk']).
        If name_emphasis is > 1, any person and entity names will be repeated correspondingly
        in the list. """
    kind = t.get("k", TOK.WORD)
    if kind not in { TOK.WORD, TOK.PERSON, TOK.ENTITY }:
        # No associated stem
        return []
    if kind == TOK.WORD:
        if "m" in t:
            # Obtain the stem and the word category from the 'm' (meaning) field
            stem = t["m"][0]
            cat = t["m"][1]
            return [ (stem, cat) ]
        else:
            # Sérnafn
            stem = t["x"]
            return [ (stem, "entity") ]
    elif kind == TOK.PERSON:
        # The full person name, in nominative case, is stored in the 'v' field
        stem = t["v"]
        if "t" in t:
            # The gender is at the end of the corresponding terminal name
            gender = "_" + t["t"].split("_")[-1]
        elif "g" in t:
            # No terminal: there might be a dedicated gender ('g') field
            gender = "_" + t["g"]
        else:
            # No known gender
            gender = ""
        return [ (stem, "person" + gender) ]
    else:
        # TOK.ENTITY
        stem = t["x"]
        return [ (stem, "entity") ]
    return []

def test_2(token_stream):

    """ Parse a stream of tokens looking for phrases and making substitutions.
        Second pass
    """
    token = None
    try:
        while True:
            token = next(token_stream)
            if token.txt:
                print("T2-error:{}-{}".format(token.txt, token.error))
            yield token

    except StopIteration:
        pass
    if token:
        yield token