"""Language detection for the free-form path, standard library only.

The catalog is English, so every reader in :mod:`src.ground` works on English
text. A shopper who writes in another language is detected here and the
message is translated by the model before the catalog reads it; the reply is
then phrased back in the shopper's language. Nothing in this module runs on
the organizer's protocol wording, which the template parser consumes first.

Detection is deliberately conservative: a script that is not Latin decides
immediately; a Latin-script language needs at least two of its function
words and more of them than English function words, so an English sentence
with one foreign word, a brand name or a place stays English (no
translation call, no tokens).
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)

# Unicode block -> language code. Han without kana is read as Chinese.
_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x3040, 0x30FF, "ja"),   # hiragana, katakana
    (0xAC00, 0xD7AF, "ko"),   # hangul syllables
    (0x1100, 0x11FF, "ko"),
    (0x3130, 0x318F, "ko"),
    (0x0E00, 0x0E7F, "th"),
    (0x0E80, 0x0EFF, "lo"),
    (0x1780, 0x17FF, "km"),
    (0x1000, 0x109F, "my"),
    (0x0600, 0x06FF, "ar"),
    (0x0750, 0x077F, "ar"),
    (0x0590, 0x05FF, "he"),
    (0x0400, 0x04FF, "ru"),
    (0x0370, 0x03FF, "el"),
    (0x0900, 0x097F, "hi"),
    (0x0980, 0x09FF, "bn"),
    (0x0B80, 0x0BFF, "ta"),
    (0x4E00, 0x9FFF, "zh"),
    (0x3400, 0x4DBF, "zh"),
    (0xF900, 0xFAFF, "zh"),
)

# Letters that occur in Vietnamese orthography and in no other common
# Latin-script language: the horn and breve vowels, the barred d, and the
# vowels carrying a tone mark.
_VIETNAMESE_LETTERS = frozenset(
    "ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
)

# Function and everyday shopping words that are not English words. Words
# shared with English ("may", "do", "that", "color", "bag") are left out on
# purpose; a single hit never decides.
_FUNCTION_WORDS: dict[str, frozenset[str]] = {
    "id": frozenset("""
        saya aku mau ingin nak hendak mahu cari mencari carikan beli membeli
        yang untuk dengan dan tidak nggak gak enggak ada ini itu warna harga
        murah bahan ukuran sepatu baju celana jaket kemeja kaos rok gaun
        kasut seluar beg dompet ikat pinggang kulit hitam putih merah biru
        bawah dari lebih kurang bisa boleh tolong bagus saja sekitar kira
        """.split()),
    "tl": frozenset("""
        ako gusto ng na para ang mga kulay presyo mura sapatos damit
        pantalon relo sinturon kailangan hanapin naghahanap bili bibili
        itim puti pula asul ito iyan wala meron mayroon ba lang
        """.split()),
    "es": frozenset("""
        quiero busco necesito zapatos zapatillas camisa camiseta pantalones
        vestido bolso reloj cinturón cinturon cuero negro negra blanco blanca
        rojo roja azul barato barata precio menos para con una unos unas algo
        por favor tengo talla quisiera busca
        """.split()),
    "pt": frozenset("""
        quero procuro preciso sapatos tênis tenis camisa camiseta calças
        calcas vestido bolsa relógio relogio cinto couro preto preta branco
        branca vermelho vermelha azul barato barata preço preco menos para
        com uma uns umas algo favor tenho tamanho gostaria
        """.split()),
    "fr": frozenset("""
        je cherche veux voudrais besoin chaussures chemise pantalon robe sac
        montre ceinture cuir noir noire blanc blanche rouge bleu bleue pas
        cher chère prix moins pour avec une des quelque chose plaît plait
        taille aimerais
        """.split()),
    "de": frozenset("""
        ich suche möchte moechte brauche schuhe hemd hose kleid tasche uhr
        gürtel guertel leder schwarz weiß weiss rot blau günstig guenstig
        billig preis unter für fuer mit eine einen bitte größe groesse etwas
        """.split()),
    "vi": frozenset("""
        tôi toi muốn muon mua giày giay dép dep quần quan áo ao túi tui xách
        xach không khong của cua màu mau giá gia rẻ dưới duoi đồng ho ví
        thắt lưng
        """.split()),
}

_ENGLISH_WORDS = frozenset("""
    i the a an and for with want need looking something under please have
    that this are is it my me to of in on some any do does not no yes one
    first like would could should can get got about or but so if at by from
    than then there these those what which who how when where why
    """.split())

LANGUAGE_NAMES = {
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "th": "Thai",
    "lo": "Lao", "km": "Khmer", "my": "Burmese", "ar": "Arabic",
    "he": "Hebrew", "ru": "Russian", "el": "Greek", "hi": "Hindi",
    "bn": "Bengali", "ta": "Tamil", "vi": "Vietnamese",
    "id": "Indonesian or Malay", "tl": "Filipino", "es": "Spanish",
    "pt": "Portuguese", "fr": "French", "de": "German",
    "latin": "the shopper's language",
}


def _script_of(char: str) -> str | None:
    point = ord(char)
    for low, high, code in _SCRIPT_RANGES:
        if low <= point <= high:
            return code
    return None


def detect(message: str) -> str | None:
    """Language code of a message that is not English, else ``None``.

    Non-Latin scripts decide on the first letter seen (kana beats Han so
    Japanese is not read as Chinese). Latin-script languages are decided by
    function words; a message with too few of them is treated as English.
    ``"latin"`` is returned for a Latin-script message with several accented
    letters that matched no word list, so it is still translated.
    """
    if not message:
        return None
    scripts: dict[str, int] = {}
    latin = 0
    accented = 0
    vietnamese = 0
    for char in message:
        if not char.isalpha():
            continue
        code = _script_of(char)
        if code is not None:
            scripts[code] = scripts.get(code, 0) + 1
            continue
        latin += 1
        if ord(char) > 127:
            lowered = char.lower()
            if lowered in _VIETNAMESE_LETTERS:
                vietnamese += 1
            # A letter with a combining accent, in any Latin language.
            if unicodedata.decomposition(lowered):
                accented += 1
    if scripts:
        # An English sentence quoting one catalog string in another script
        # ("I just need 进口 and Pull On closure") stays English: the script
        # decides only when it carries a real share of the letters.
        total = sum(scripts.values())
        if latin == 0 or (total >= 2 and total >= 0.3 * (total + latin)):
            if "ja" in scripts:
                return "ja"
            return max(scripts, key=lambda code: (scripts[code], code))
    if vietnamese >= 2:
        return "vi"

    words = [token.lower() for token in _TOKEN.findall(message)]
    english = sum(1 for word in words if word in _ENGLISH_WORDS)
    best_code, best_hits = None, 0
    for code, vocabulary in _FUNCTION_WORDS.items():
        hits = sum(1 for word in words if word in vocabulary)
        if hits > best_hits:
            best_code, best_hits = code, hits
    if best_hits >= 2 and best_hits > english:
        return best_code
    if vietnamese >= 1 and best_code == "vi":
        return "vi"
    if accented >= 3 and english == 0:
        return "latin"
    return None


def language_name(code: str | None) -> str | None:
    if code is None:
        return None
    return LANGUAGE_NAMES.get(code, "the shopper's language")
