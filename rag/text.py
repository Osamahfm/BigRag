import re
import unicodedata

STOPWORDS = set("a an the is are was were what which how of in on at to for and or it its this that does do with from about".split())
STOPWORDS |= {"ما", "هي", "هو", "في", "من", "عن", "كيف", "هل", "و", "على"}


def tokens(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    return [t for t in re.findall(r"[^\W_]+", text, re.UNICODE) if t not in STOPWORDS]


def clean(text):
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(re.sub(r"[ \t]+", " ", line).strip()
                                                 for line in text.splitlines())).strip()
