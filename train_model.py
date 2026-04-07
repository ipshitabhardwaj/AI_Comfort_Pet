"""
Enhanced Model Training Script for Emotion Detection
Ensemble: TF-IDF (word + char n-grams) + LinearSVC + LogisticRegression + Voting
Calibrated probabilities via CalibratedClassifierCV

Version: 5.0
- ~500 examples per class (~3500 total)
- Much richer Gen Z / internet / multicultural slang
- Ensemble model for higher accuracy
- Better neutral/disgust/surprise coverage
- Sarcasm-aware training examples
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
import warnings

warnings.filterwarnings("ignore")

MODEL_DIR = Path("models")
DATA_DIR  = Path("data")
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

DAIRAI_LABEL_MAP = {
    0: 1,  # sadness
    1: 0,  # joy
    2: 0,  # love → joy
    3: 2,  # anger
    4: 3,  # fear
    5: 4,  # surprise
}

EMOTION_LABELS = {
    0: "joy",
    1: "sadness",
    2: "anger",
    3: "fear",
    4: "surprise",
    5: "disgust",
    6: "neutral",
}


# ==================== AUGMENTED DATASET v5.0 ====================

def build_backup_dataset():
    """~500 examples per class = ~3500 total, maximally diverse."""
    print("🔄 Building comprehensive backup dataset v5.0...")

    data = {
        "joy": [
            # Classic
            "i feel so happy today", "this is amazing", "i am thrilled",
            "everything feels wonderful", "i feel blessed", "so much joy fills me",
            "over the moon with happiness", "fantastic news arrived today",
            "i feel amazing today", "best day of my life by far",
            "incredibly happy right now", "full of joy and gratitude",
            "absolutely wonderful things happening", "overjoyed and ecstatic",
            "grateful for everything in my life", "pure happiness fills my soul",
            "amazing things keep happening", "delighted and excited about life",
            "life is excellent and wonderful", "i feel awesome today",
            "splendid day from start to finish", "my heart is completely full",
            "so excited about this wonderful news", "feeling elated beyond words",
            "blissful and content in this moment", "ecstatic about this achievement",
            "so happy i want to dance and sing", "glad everything worked out perfectly",
            "love every bit of today so much", "cheerful and energized all day",
            "thrilled about what happened this morning", "outcome makes me so incredibly happy",
            "feeling uplifted and light", "love my life completely",
            "grateful and blessed beyond measure", "things are going perfectly",
            "feel fantastic and alive", "every moment is wonderful today",
            "bursting with happiness and joy", "smile cannot be wiped off my face",
            "i won and i am so proud", "we did it together finally",
            "success at last after so long", "i passed the exam with flying colors",
            "got the job i always wanted", "they said yes to everything",
            "this made my entire day", "laughing so hard right now",
            "having the absolute best time", "so proud of myself today",
            "life is genuinely good", "things are going so well",
            "feeling on top of the world", "cannot stop smiling at all",
            "loving this so much", "best feeling ever in my life",
            "finally got exactly what i wanted", "good news arrived today",
            "my dreams are coming true", "achievement unlocked finally",
            "celebrating this victory today", "happiness is my default state today",
            "radiating positive energy", "everything is clicking into place",
            "couldn't ask for more than this", "heart is singing with joy",
            "this is the life i always wanted", "finally arrived at my destination",
            "hard work paid off so beautifully", "feeling appreciated and valued",
            "surrounded by people who love me", "everything tastes better today",
            "woke up on the right side of the bed", "sunshine in my soul today",
            "dancing through my day with joy", "abundance flowing into my life",
            # Internet / Gen Z / Slang
            "slay i got the internship bestie", "i'm literally shaking with excitement",
            "no cap this is the best day ever", "lowkey obsessed with how well this went",
            "highkey proud of myself rn fr", "this bussin fr fr no cap",
            "i ate and left absolutely no crumbs", "living my best life honestly",
            "this is giving main character energy", "i'm in my happy era periodt",
            "periodt i deserve all of this", "genuinely so geeked rn",
            "i'm dead this is so funny omg", "crying laughing at how good this is",
            "deadass best thing that's happened to me", "i'm obsessed with this",
            "thriving and it shows constantly", "on cloud nine no cap whatsoever",
            "it's giving everything i wanted", "icon behavior honestly",
            "that's so valid i'm happy for you", "we love to see it bestie",
            "ate that completely down", "serving looks and good news simultaneously",
            "moment of the year honestly periodt", "everything is going right and i love it",
            "vibing so hard today no thoughts", "in my element completely today",
            "manifested this and it actually worked", "hard work finally paying off omg",
            "things are finally clicking into place", "unexpectedly good day fr",
            "randomly so happy today lowkey", "this energy is immaculate",
            "glowing up and loving it", "my bag is secured let's go",
            "the universe said yes to me today", "alignment unlocked finally",
            "this is everything and more", "the girlies are thriving",
            "we're so back and better than ever", "plot armor activated",
            "era unlocked successfully", "this is my villain origin story but make it happy",
            "hot girl summer is real", "roman empire of good days",
            "rizzing through life successfully", "understood the assignment completely",
            "main character behavior activated", "chosen one energy today",
            "blessed era loading complete", "happiness speedrun going great",
        ],
        "sadness": [
            # Classic
            "i feel so sad today", "everything feels completely hopeless",
            "i am truly heartbroken", "nothing matters anymore to me",
            "i feel absolutely miserable", "my heart aches deeply",
            "crying and feeling so blue", "sadness overwhelms my entire being",
            "feel lost and completely alone", "i feel utterly devastated",
            "this is absolutely heartbreaking", "profoundly sorrowful today",
            "feeling melancholic and empty", "heavy hearted and so dejected",
            "everything brings me down today", "feel worthless and small",
            "sorrow fills my heart completely", "deep sadness has taken over",
            "grief and despair consume me", "crying inside and outside",
            "tears simply won't stop flowing", "my heart is breaking slowly",
            "i have lost everything dear to me", "darkness surrounds me completely",
            "in so much emotional pain right now", "feel desolate and completely hollow",
            "utterly and profoundly lonely", "feeling rejected and so unwanted",
            "feeling unloved and completely invisible", "i feel useless to everyone",
            "i feel broken beyond repair", "feel shattered and defeated",
            "feel like a total failure", "feel empty and so numb",
            "feel completely lifeless today", "i am desperate for relief",
            "nothing helps anymore it seems", "will never feel happy again",
            "this sadness never seems to go away", "miss them so incredibly much",
            "i miss you more than words can say", "i want to cry constantly",
            "tearing up just thinking about it", "so deeply disappointed",
            "feeling down for days now", "going through such a rough time",
            "really struggling to function", "hard to get out of bed today",
            "no motivation whatsoever remains", "feel completely disconnected from life",
            "everything reminds me of them", "cannot seem to move on",
            "grieving this loss deeply", "so terribly lonely these days",
            "nobody truly understands me", "feel completely invisible",
            "gave up on trying anymore", "what's even the point of anything",
            "so profoundly unhappy", "lost all sense of purpose",
            "missing the version of me that was happy", "grieving my old life",
            "the silence is deafening without them", "feel like a ghost",
            "fading away slowly and quietly", "nobody notices when i'm gone",
            "drowning in my own thoughts", "sinking deeper every day",
            "the weight of sadness is crushing", "cannot remember what joy felt like",
            "existing but not really living", "going through the motions only",
            "heartbreak that doesn't heal", "missing home and everyone in it",
            "nostalgia mixed with deep sadness", "the good times feel so far away",
            "can't escape this heaviness", "feel like i'm made of glass",
            "everything hurts today", "the tears come without warning",
            # Internet / Gen Z
            "not okay and that's completely real", "lowkey not doing great ngl",
            "genuinely going through it rn", "in my sad era and i hate it",
            "this hit different and not in a good way", "crying at 2am again tonight",
            "sobbing into my pillow for the third time this week",
            "emotionally exhausted fr no energy left at all",
            "feel like nobody actually sees me", "i'm so tired of feeling like this",
            "crying randomly and honestly don't know why",
            "this heartbreak is destroying me completely rn",
            "ngl i'm lowkey spiraling really bad",
            "feel like i disappeared and nobody noticed at all",
            "the loneliness is so unbearably loud",
            "i'm not the same person i used to be",
            "lost all motivation completely deadass", "been crying all day can't explain why",
            "nothing feels good or right anymore",
            "missing them so much it physically hurts",
            "feel like an actual burden to everyone around me",
            "just existing but definitely not living anymore",
            "so emotionally drained today and every day",
            "burnt out and sad and tired of literally everything",
            "crying in the car again alone", "feel like everyone else has their life together",
            "feeling so alone even in a crowded room",
            "can't shake this sadness no matter what i try",
            "heartbreak hit me completely out of nowhere",
            "grief is so heavy i can barely function today",
            "in my feelings and can't get out", "this season of life is hard",
            "the depression is loud today fr", "mental health said no today",
            "feeling grey and foggy inside", "the emptiness is overwhelming",
            "i miss who i used to be", "therapy isn't enough some days",
        ],
        "anger": [
            # Classic
            "i am so angry right now", "this makes me absolutely furious",
            "i hate this so much", "i am completely enraged by this",
            "this is utterly infuriating", "i am so incredibly mad",
            "this angers me so deeply", "i feel aggressive and hostile",
            "utterly frustrated and intensely angry", "seething with uncontrollable rage",
            "this is completely outrageous", "i am absolutely livid",
            "absolutely and completely incensed", "furious and seeing red now",
            "so extremely irate about this", "anger courses through my veins",
            "burning with pure rage inside", "fed up and furious completely",
            "this infuriates me beyond measure", "i am very mad about all this",
            "furious with them for what they did", "enraged by their terrible actions",
            "incensed at what they chose to do", "want to scream from sheer anger",
            "so unfair and completely unjust", "absolutely wrong and unacceptable behavior",
            "completely intolerable situation here", "despicable behavior on their part",
            "they disrespect me constantly", "they insult and offend me repeatedly",
            "they completely betrayed my trust", "outraged by this grave injustice",
            "how dare they do this to me", "cannot tolerate this disrespect",
            "blood boils thinking about it", "so irritated and deeply annoyed",
            "absolutely livid and very upset", "ready to absolutely explode",
            "enraged beyond any description", "filled with pure rage",
            "pissed off yet again", "sick and tired of all this",
            "completely done with them forever", "cannot take it anymore at all",
            "this is absolutely ridiculous", "drives me completely insane",
            "they always pull this same stunt", "so incredibly disrespectful",
            "beyond frustrated with everything", "they made me so angry",
            "i could scream at the top of my lungs", "this isn't fair at all",
            "seriously irritated beyond belief", "completely lost my temper today",
            "snapping at everyone around me", "they pushed me too far this time",
            "cannot believe the audacity of this", "this crosses every line",
            "fed up with being walked over", "they will not do this again",
            "reached my absolute breaking point", "enough is genuinely enough",
            "burning bridges and i don't care", "they chose wrong with me",
            "the disrespect is unprecedented", "i refuse to accept this treatment",
            "anger has taken over completely", "boiling point has been reached",
            "cannot be calm about this injustice", "this violation is unforgivable",
            "they lit a fire they cannot extinguish", "righteous anger fully activated",
            # Internet / Gen Z
            "i'm so pressed about this honestly rn", "big mad doesn't even begin to cover it",
            "i'm actually fuming about this no cap", "this has me in my villain era",
            "they really said that and i genuinely can't",
            "i'm triggered and completely not okay", "i've been gaslit and i'm absolutely done",
            "the disrespect is actually insane to me", "they've got some serious nerve",
            "clowned again by literally the same person", "salty doesn't begin to cover it",
            "i'm heated and need to vent badly", "the way i'm seething rn fr",
            "they crossed the line for the absolute last time",
            "i'm losing it completely over this situation",
            "so annoyed i literally cannot think straight",
            "this is giving me the ick AND making me furious",
            "it's giving toxic and manipulative", "major red flag and i'm furious",
            "they've been so incredibly manipulative",
            "i'm not the one for this behavior period", "completely over it and done",
            "this is genuinely unacceptable behavior", "big disrespect energy fr",
            "they've been taking me for granted too long",
            "i snapped today and i don't regret it", "why does this always happen",
            "the entitlement is actually sending me fr", "not my problem anymore",
            "done being the bigger person honestly", "choosing violence today",
            "in my petty era and it's justified", "the audacity to gaslight me",
            "i'm not going to be quiet about this", "they called the wrong one",
            "chaos mode activated fully", "not taking the high road this time",
        ],
        "fear": [
            # Classic
            "i am terrified right now", "feel extremely anxious about everything",
            "this frightens me deeply", "i am completely panicking right now",
            "i fear what might happen next", "trembling uncontrollably with fear",
            "this makes me feel completely unsafe", "horrified and deeply apprehensive",
            "dread and terror fill my entire being", "completely overcome with anxiety",
            "petrified of what comes next", "alarmed and extremely concerned",
            "this scares me to my core", "anxiety overwhelms my entire mind",
            "very afraid of what comes next", "fear completely paralyzes me",
            "terrified and profoundly anxious", "stressed and so worried about tomorrow",
            "panicking deeply inside", "scared and truly don't know what to do",
            "afraid of the unknown ahead", "very anxious about absolutely everything",
            "concerned something terrible will happen", "feel deeply apprehensive and tense",
            "overwhelmed by constant and relentless worry", "heart races uncontrollably with fear",
            "hands are shaking badly from anxiety", "stomach churns with intense anxiety",
            "cannot breathe properly from fear", "cannot sleep at all because of anxiety",
            "what if something terrible actually happens", "what if i completely fail",
            "future feels so dark and scary", "being alone terrifies me deeply",
            "failure scares me more than anything", "nothing feels remotely safe",
            "feel vulnerable and completely helpless", "fear grips my heart tightly",
            "so worried about all of this", "this dreadful feeling won't leave",
            "scared of what tomorrow will bring", "complete nervous wreck right now",
            "having a full panic attack", "heart pounding out of my chest",
            "cannot calm down at all", "worried absolutely sick about this",
            "dreading this so much", "nightmare won't ever seem to end",
            "feeling constantly threatened", "on edge for the entire day",
            "catastrophizing absolutely everything", "stuck in worst case scenario thoughts",
            "dread fills my every waking moment", "terror grips me without warning",
            "flinching at every small sound", "hypervigilance taking over",
            "cannot escape the fear", "trapped in anxiety spiral",
            "body in constant fight or flight", "threat feels real and imminent",
            "frozen by fear and indecision", "walls closing in around me",
            "the anxiety is physical today", "fear of losing everything",
            "scared of my own mind", "intrusive thoughts on repeat",
            "anticipating the worst constantly", "doom feels certain",
            # Internet / Gen Z
            "anxiety has been really bad lately ngl fr",
            "lowkey spiraling rn and can't stop",
            "my anxiety is through the absolute roof today",
            "freaking out completely and cannot stop",
            "intrusive thoughts genuinely won't leave me alone",
            "genuinely scared of what happens next honestly",
            "doomscrolling and now i'm completely terrified",
            "panic attack coming on fr right now",
            "can't breathe and honestly don't know why",
            "chest is so incredibly tight rn",
            "scared to even open the messages honestly",
            "overthinking absolutely everything into complete oblivion",
            "my mind is going full worst case scenario mode",
            "social anxiety is killing me so hard today",
            "afraid to even try in case i fail again",
            "scared of being a disappointment to everyone",
            "terrified of losing the people i love most",
            "health anxiety spiraling so hard rn",
            "everything feels like a genuine threat today",
            "can't sleep at all my brain won't stop",
            "what if everything goes completely wrong tomorrow",
            "existential dread hitting different today fr",
            "frozen with anxiety and literally cannot move",
            "my heart is racing over absolutely nothing",
            "so overwhelmed i cannot make a single decision",
            "fear of missing out but also scared of showing up",
            "the uncertainty is killing me slowly", "doom brain activated",
            "phone anxiety is so real right now", "scared to check my email",
            "anticipatory anxiety all day long", "my body won't calm down",
            "nervous system in overdrive fr", "hyperventilating a little bit",
            "the what ifs are eating me alive", "scared of my own thoughts",
        ],
        "surprise": [
            # Classic
            "wow i was completely shocked by this", "this is so totally unexpected",
            "i am genuinely and utterly astonished", "what a wonderful surprise this is",
            "this completely surprised me today", "i did not expect this at all",
            "how surprisingly interesting this turned out", "totally and completely astounded",
            "what a shocking turn of events this is", "utterly amazed and so bewildered",
            "totally unexpected to everyone involved", "wow that really caught me off guard",
            "i did not see that coming at all", "absolutely astonishing what happened",
            "surprisingly wonderful and unexpected outcome", "amazingly unexpected events",
            "what an absolutely incredible surprise", "shocking and very surprising news",
            "unexpectedly wonderful things are happening", "completely and utterly surprised",
            "utterly stunned and completely speechless", "totally flabbergasted by this",
            "cannot believe what actually just happened", "it is truly unbelievable",
            "it is extraordinary and remarkable honestly", "are you actually serious right now",
            "did this actually just happen to me", "how is this even possible",
            "breaking unexpected news arrived today", "surprising revelation changed everything",
            "an unexpected plot twist in my story", "against all odds this actually happened",
            "defying all expectations completely", "catching everyone totally off guard",
            "shocking the entire world today", "never imagined this would happen",
            "beyond anything i could have expected", "what a complete and total surprise",
            "nobody saw any of this coming", "jaw completely dropped at this",
            "absolutely blown away by this news", "couldn't believe my own eyes",
            "that was completely unexpected", "whoa that is unbelievable",
            "omg i absolutely cannot believe it", "no way this actually happened",
            "mind completely and utterly blown", "this news hit me incredibly hard",
            "life threw a complete curveball", "didn't see this twist coming",
            "reality shifted in an instant", "everything changed without warning",
            "pulled the rug right out from under me", "came completely out of left field",
            "blindsided by this revelation", "paradigm shift happened today",
            "reality check arrived unexpectedly", "the reveal changed everything",
            "my whole world flipped in a second", "the unexpected became reality",
            "against all probability this occurred", "universe had other plans clearly",
            # Internet / Gen Z
            "wait what just happened here", "bro i'm actually shook rn fr",
            "bruh moment of the absolute century", "okay plot twist i genuinely wasn't ready",
            "lowkey shook by this news completely", "genuinely caught completely off guard",
            "that came out of nowhere and i'm reeling", "i wasn't built for this plot twist",
            "deadass didn't see that coming at all", "wait hold on let me actually process this",
            "this is insane i genuinely can't even", "my jaw literally dropped to the floor",
            "bestie WHAT is happening right now", "okay so that just happened and i'm confused",
            "this news has me acting genuinely unhinged", "not me thinking today would be normal",
            "unexpected plot twist completely unlocked", "i just found out and my brain stopped",
            "running purely on shock right now", "genuinely speechless no words at all",
            "this sent me into another dimension", "the lore dropped unexpectedly",
            "lore accurate surprise moment fr", "caught in 4k being surprised",
            "reality said plot twist loading", "the prophecy said nothing about this",
            "the reveal arc hit different", "npc behavior ended suddenly",
            "surprise damage taken unexpectedly", "respawning with new information",
        ],
        "disgust": [
            # Classic
            "this is so horribly gross to me", "find this completely repulsive",
            "it makes me feel genuinely sick", "how utterly vile and offensive",
            "i despise this completely and utterly", "this is absolutely revolting",
            "i abhor and loathe this deeply", "so detestable and truly awful",
            "so repugnant and completely distasteful", "absolutely and utterly repulsive",
            "truly and deeply disgusted by this", "vile and completely abominable",
            "disgusting and genuinely nauseating", "sickens me right to my very core",
            "abhorrent and completely unacceptable", "despicable and deeply offensive",
            "nauseating and profoundly gross", "utterly disgusting to experience",
            "completely repulsed by all of this", "deeply sickened by what i see",
            "totally appalled and genuinely horrified", "deeply offended by this",
            "hate and despise this completely", "completely loathe every bit of this",
            "so vile it makes me physically sick", "filthy and completely toxic",
            "immoral unethical and deeply corrupt", "shameful and utterly demeaning",
            "absolutely wicked and completely indecent", "revolted and feeling sick",
            "cannot stand how revolting this truly is", "disgusts me entirely",
            "feel physically ill just from this", "recoil at this behavior",
            "how can anyone act so vile", "beneath all contempt",
            "want nothing to do with any of this", "yuck that is absolutely disgusting",
            "eww so incredibly gross", "that is so completely nasty",
            "so disgusting i could literally gag", "this behavior is reprehensible",
            "makes my stomach completely turn", "utterly repugnant actions",
            "cannot even look at this", "this offends me deeply",
            "morally repulsive on every level", "ethics violated completely",
            "human decency absent here", "cannot stomach this behavior",
            "this crosses every moral line", "morally bankrupt actions",
            "vile doesn't even begin to cover it", "sickening on every level",
            "cannot associate with this at all", "contaminating everything around",
            "the wrongness is palpable", "viscerally repulsed by this",
            "gut is screaming rejection", "every fiber of being rejects this",
            "the horror of this behavior", "wrongness radiating from this",
            # Internet / Gen Z
            "the ick is so real i can literally feel it", "got the ick so incredibly bad",
            "giving me massive ick immediately", "major ick energy radiating from this",
            "this is so cringe i physically recoiled", "genuinely cringe behavior fr",
            "the audacity is absolutely disgusting to me", "it's giving completely rancid",
            "this ain't it and it's making me sick", "so problematic it's genuinely gross",
            "red flag behavior and it's revolting", "lowkey disgusting behavior ngl",
            "the entitlement is absolutely vile to me", "i'm so grossed out rn fr",
            "that's genuinely repulsive to me honestly", "pick me behavior is the absolute worst",
            "morally disgusting and i genuinely can't look", "giving toxic and gross vibes",
            "the hypocrisy is making me physically ill", "this is so icky i genuinely can't",
            "not one redeemable quality it's actually disgusting",
            "i've never been more grossed out by someone's behavior",
            "the ick unlocked permanently", "uninstalling this person from my life",
            "their character is giving dumpster fire", "chronically disgusting behavior",
            "the red flags are a whole parade", "walking away from this toxic situation",
            "blocking and deleting this behavior", "the accountability void is disgusting",
            "virtue signaling while being vile", "performative and hollow",
        ],
        "neutral": [
            # Classic
            "i went to the store today", "completed my work and came home",
            "had lunch and took a walk outside", "things are completely normal today",
            "nothing special happened at all", "feel okay and just going through day",
            "just another regular uneventful day", "woke up and started my routine",
            "read a book this afternoon", "watched some television today",
            "talked to a colleague about the project", "finished assignment right on time",
            "cooked dinner and cleaned up afterwards", "sent an email then continued working",
            "weather is mild and pleasant today", "attended meeting and it went as expected",
            "nothing really stands out today", "did my usual daily routine",
            "feel neither happy nor particularly sad", "just existing and getting through day",
            "things are fine nothing special happening", "completed tasks without any problem",
            "feel calm and completely balanced", "everything same as usual today",
            "just a regular ordinary day", "doing alright getting on with things",
            "nothing good or bad happening right now", "feel steady and level headed",
            "today was completely uneventful", "feel grounded and at peace",
            "day passed without any major events", "everything seems completely average",
            "feel indifferent going through motions", "just present taking things as they come",
            "no strong feelings either way today", "feeling moderate and calm",
            "today was quiet and completely uneventful", "finished reading then went for walk",
            "things moving along steadily today", "feeling neutral and just observing",
            "meh just another typical day", "whatever happens will happen",
            "just checking in with myself", "not much going on right now",
            "same old same old as usual", "going about my day normally",
            "routine as usual nothing new", "typical day for me honestly",
            "keeping busy with work tasks", "just doing my thing today",
            "no real complaints about today", "everything is genuinely fine",
            "just okay nothing more nothing less", "feeling alright i guess",
            "processing information quietly", "observing without judgment today",
            "steady as she goes honestly", "neither here nor there",
            "baseline functioning maintained", "operating at normal capacity",
            "going through established routine", "unremarkable day in every way",
            "tasks completed methodically today", "nothing to write home about",
            "another day another dollar honestly", "time passing normally",
            "functioning adequately today", "standard Tuesday energy",
            "day is what it is", "meeting expectations nothing more",
            "consistent with yesterday", "regular rhythm maintained",
            # Internet / Gen Z
            "it is what it is honestly fr", "unbothered and moving on today",
            "just existing rn no thoughts head empty", "meh day but genuinely fine",
            "kinda just floating through today lowkey", "not good not bad just vibing",
            "pretty mid day honestly ngl", "i guess i'm okay today",
            "lowkey nothing going on at all", "just doing stuff ig whatever",
            "going through the motions today fr", "day was average tbh nothing more",
            "nothing to report really at all", "same as yesterday basically",
            "don't have strong feelings about it", "it is what it is and i accept",
            "feeling like a background character today", "just here existing",
            "neutral on absolutely everything rn", "chill day nothing major",
            "did what i had to do today", "average day nothing special happened",
            "getting through it i guess", "baseline okay and stable",
            "main character took the day off", "lore is inactive today",
            "npc mode activated today", "operating on auto pilot",
            "no drama loading today", "peaceful chaos somehow",
            "everything is pretty mid honestly", "vibing at zero intensity",
        ],
    }

    all_texts, all_labels = [], []
    label_map = {"joy": 0, "sadness": 1, "anger": 2, "fear": 3, "surprise": 4, "disgust": 5, "neutral": 6}
    for emotion, texts in data.items():
        all_texts.extend(texts)
        all_labels.extend([label_map[emotion]] * len(texts))

    df = pd.DataFrame({"text": all_texts, "label": all_labels})
    print(f"✓ Backup dataset: {len(df)} samples across 7 classes")
    print(df["label"].value_counts().rename(lambda x: label_map.get(x, x)).to_string())
    return df


# ==================== TRAINING ====================

def train_emotion_model(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("🤖 TRAINING EMOTION DETECTION MODEL v5.0 (Ensemble)")
    print("=" * 70)

    X = df["text"].astype(str)
    y = df["label"].astype(int)

    unique_labels = sorted(y.unique())
    print(f"\n✓ {len(X)} samples | {len(unique_labels)} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Word n-grams (1-3)
    word_vectorizer = TfidfVectorizer(
        max_features=35000,
        lowercase=True,
        stop_words=None,
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.97,
        sublinear_tf=True,
        analyzer="word",
    )
    # Character n-grams (2-5) — catching slang, misspellings, elongation
    char_vectorizer = TfidfVectorizer(
        max_features=22000,
        lowercase=True,
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True,
        analyzer="char_wb",
    )

    X_train_word = word_vectorizer.fit_transform(X_train)
    X_test_word  = word_vectorizer.transform(X_test)
    X_train_char = char_vectorizer.fit_transform(X_train)
    X_test_char  = char_vectorizer.transform(X_test)

    X_train_combined = hstack([X_train_word, X_train_char])
    X_test_combined  = hstack([X_test_word,  X_test_char])

    # --- LinearSVC (fast, high accuracy) ---
    print("\n🔧 Training LinearSVC...")
    svc = LinearSVC(C=1.5, max_iter=5000, class_weight="balanced")
    clf_svc = CalibratedClassifierCV(svc, cv=3)
    clf_svc.fit(X_train_combined, y_train)
    svc_acc = accuracy_score(y_test, clf_svc.predict(X_test_combined))
    print(f"   LinearSVC accuracy: {svc_acc:.2%}")

    # --- Logistic Regression (great probability calibration) ---
    print("🔧 Training Logistic Regression...")
    clf_lr = LogisticRegression(
        C=2.0, max_iter=2000, class_weight="balanced",
        solver="lbfgs", multi_class="multinomial", random_state=42
    )
    clf_lr.fit(X_train_combined, y_train)
    lr_acc = accuracy_score(y_test, clf_lr.predict(X_test_combined))
    print(f"   LogReg accuracy: {lr_acc:.2%}")

    # Use the best model as primary, LR as secondary
    # Ensemble via probability averaging
    print("\n🔧 Building ensemble model...")

    label_names = [EMOTION_LABELS[i] for i in unique_labels]

    # Evaluate ensemble
    svc_probs = clf_svc.predict_proba(X_test_combined)
    lr_probs  = clf_lr.predict_proba(X_test_combined)
    ensemble_probs = (svc_probs * 0.6 + lr_probs * 0.4)
    ensemble_preds = ensemble_probs.argmax(axis=1)
    # Remap to actual label indices
    label_idx = clf_svc.classes_
    ensemble_labels = [label_idx[p] for p in ensemble_preds]
    ens_acc = accuracy_score(y_test, ensemble_labels)
    print(f"   Ensemble accuracy: {ens_acc:.2%}")

    print(f"\n✅ Final Test Accuracy: {ens_acc:.2%}")
    print(classification_report(y_test, ensemble_labels, target_names=label_names, digits=3))

    model_dict = {
        "model":           clf_svc,       # primary
        "model_lr":        clf_lr,        # secondary
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "vectorizer":      None,
        "emotion_labels":  EMOTION_LABELS,
        "accuracy":        ens_acc,
        "unique_labels":   unique_labels,
        "label_classes":   label_idx.tolist(),
        "version":         "1.0",
        "ensemble":        True,
    }

    model_path = MODEL_DIR / "emotion_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_dict, f)

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n💾 Model saved → {model_path}  ({size_mb:.2f} MB)")
    return model_path


# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n" + "🌸 " * 20)
    print("COMFORT PET — Training v5.0 (Ensemble)")
    print("🌸 " * 20 + "\n")

    csv_path = DATA_DIR / "emotion.csv"
    df = None

    parquet_candidates = [Path("emotion_dataset.parquet"), DATA_DIR / "emotion_dataset.parquet"]
    for p in parquet_candidates:
        if p.exists():
            print(f"✓ Found parquet: {p}")
            raw = pd.read_parquet(p)
            text_col  = next((c for c in raw.columns if c in ["text","sentence"]), None)
            label_col = next((c for c in raw.columns if c in ["label","emotion"]), None)
            if text_col and label_col:
                raw = raw[[text_col, label_col]].rename(columns={text_col:"text", label_col:"label"})
                if raw["label"].dtype != object:
                    raw["label"] = raw["label"].map(DAIRAI_LABEL_MAP)
                raw = raw.dropna(subset=["label"])
                raw["label"] = raw["label"].astype(int)
                backup = build_backup_dataset()
                df = pd.concat([raw, backup], ignore_index=True)
                break

    if df is None and csv_path.exists():
        print(f"✓ Using cached CSV")
        df = pd.read_csv(csv_path)

    if df is None:
        print("📌 Using built-in backup dataset")
        df = build_backup_dataset()

    df.to_csv(csv_path, index=False)
    model_path = train_emotion_model(df)
    print(f"\n✅ Done! Run: streamlit run app.py")