import os
import jpype
import jpype.imports
#from jpype.types import *
from transformers import AutoTokenizer

here = os.path.dirname(os.path.abspath(__file__))
jar = os.path.join(here, "zemberek-full.jar")
#print("jar found:", os.path.exists(jar))

jpype.startJVM("--enable-native-access=ALL-UNNAMED", "--sun-misc-unsafe-memory-access=allow", classpath=[jar])
jpype.imports.registerDomain("zemberek")
from zemberek.morphology import TurkishMorphology
morph = TurkishMorphology.createWithDefaults()
'''
word = "evleri"
zemex = morph.analyze(word)
results = zemex.getAnalysisResults()
print(len(results))

for analysis in results:
    surfaces = []
    morphemes = []
    for md in analysis.getMorphemeDataList():
        surfaces.append(str(md.surface))
        morphemes.append(str(md.morpheme.id))
    print(analysis.getDictionaryItem(), surfaces, morphemes)

for analysis in results:
    boundary = set()
    position = 0
    for md in analysis.getMorphemeDataList():
        position += len(str(md.surface))
        if position != len(word): boundary.add(position)
    print(sorted(boundary))
'''
def zemberek_boundaries(word):
    results = morph.analyze(word).getAnalysisResults()
    analysis = results[0]
    boundary = set()
    position = 0
    for md in analysis.getMorphemeDataList():
        position += len(str(md.surface))
        if 0 < position < len(word): boundary.add(position)
    return boundary

gpt2 = AutoTokenizer.from_pretrained("gpt2") #no Turkish, small
poro34b = AutoTokenizer.from_pretrained("LumiOpen/Poro-34B") #no Turkish, medium
bloom = AutoTokenizer.from_pretrained("bigscience/bloom-560m") #no Turkish, large
mgpt = AutoTokenizer.from_pretrained("ai-forever/mGPT") #has Turkish, small
llama3 = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct") #has Turkish, medium
gemma = AutoTokenizer.from_pretrained("google/gemma-2-2b") #has Turkish, large
roberta_tr = AutoTokenizer.from_pretrained("burakaytan/roberta-base-turkish-uncased") #full Turkish, small
turkish_bpe_128k = AutoTokenizer.from_pretrained( 
    "erenyanic/turkish_bpe_128k", #full Turkish, medium
    subfolder="tokenizer"
) 
ts_bpe_256k = AutoTokenizer.from_pretrained("tahaenesaslanturk/ts-corpus-bpe-256k-cased") #full Turkish, large

tokenizers = {"gpt2": gpt2, "poro34b": poro34b, "bloom": bloom, "mgpt": mgpt, "llama3": llama3, "gemma": gemma, "roberta_tr": roberta_tr, "turkish_bpe_128k": turkish_bpe_128k, "ts_bpe_256k": ts_bpe_256k}

#for name, tokenizer in tokenizers.items():
    #print(name, type(tokenizer.backend_tokenizer.model).__name__)

words = ["kedi", "kalem", "defterler", "pencerelerden", "gitmeyecekler", "konuşamayacağım", "sorumsuzluğunuzdan"]
#tokenised_words = []

def tokenizer_boundaries(tokenizer, word):
    encoding = tokenizer(word, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    boundary = set()
    for start, end in offsets:
        if 0 < end < len(word):
            boundary.add(end)
    return boundary

gold = {}
for word in words:
    gold[word] = zemberek_boundaries(word)

for name, tokenizer in tokenizers.items():
    matches = 0
    tok_cuts = 0
    gold_cuts = 0
    for word in words:
        pred = tokenizer_boundaries(tokenizer, word)
        matches += len(gold[word] & pred)
        tok_cuts += len(pred)
        gold_cuts += len(gold[word])
    precision = matches / tok_cuts
    recall = matches / gold_cuts
    #boundary_f1 = 2 * matches / (tok_cuts + gold_cuts) #ruled out 
    boundary_f05 = (1 + 0.25) * precision * recall / (0.25 * precision + recall)
    print(name, precision, recall, boundary_f05)
'''
for name, tokenizer in tokenizers.items():
    for word in words:
        encoding = tokenizer(word, return_offsets_mapping=True, add_special_tokens=False)
        tokenised_words.append(encoding)
        offsets = encoding["offset_mapping"]
        print(name, offsets)
        pieces = []
        for start, end in offsets:
            pieces.append(word[start:end])
        print(name, pieces)
        joined = "".join(pieces)
        print(name, joined == word)
    tokenised_words = []
'''
jpype.shutdownJVM()
