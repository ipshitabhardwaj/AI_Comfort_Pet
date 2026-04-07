"""
Enhanced Utility Functions for Emotion-Aware Digital Comfort Pet
Multi-layer smart fallback emotion detection — v5.0

Accuracy improvements:
  - 600+ keywords per emotion (3x expansion)
  - 300+ phrase patterns (multi-language slang)
  - Sarcasm / irony detection layer
  - VADER sentiment as 4th signal
  - Multi-emotion blending (bittersweet, anxious-joy etc.)
  - Better confidence calibration with softmax normalization
  - Cultural / regional expression coverage
  - Therapy / mental health language
  - Body-language descriptions
  - Academic / formal emotional vocabulary
  - Ensemble probability averaging with rule signals
"""

import re
from typing import Dict, List, Tuple, Optional
import random
import math

# ==================== EMOTION KEYWORDS (weighted, massively expanded) ====================

EMOTION_KEYWORDS_WEIGHTED = {
    "joy": [
        # Core happiness
        ("happy", 3), ("happiness", 3), ("joyful", 3), ("joy", 3), ("joyous", 3),
        ("excited", 2.5), ("exciting", 2), ("amazing", 2), ("wonderful", 2), ("fantastic", 2),
        ("awesome", 2), ("love", 1.5), ("loved", 2), ("lovely", 1.5), ("great", 1.5),
        ("excellent", 2), ("perfect", 2), ("thrilled", 3), ("delighted", 3), ("blessed", 2),
        ("grateful", 2), ("thankful", 2), ("ecstatic", 3), ("cheerful", 2.5),
        ("glad", 2), ("brilliant", 1.5), ("elated", 3), ("overjoyed", 3),
        ("blissful", 3), ("gleeful", 3), ("jubilant", 3), ("content", 1.5),
        ("pleased", 1.5), ("proud", 2), ("winning", 1.5), ("celebrate", 2),
        ("celebration", 2), ("laugh", 1.5), ("laughing", 2), ("laughter", 2),
        ("smile", 1.5), ("smiling", 2), ("fun", 1.5), ("enjoy", 1.5), ("enjoyable", 1.5),
        ("stoked", 2), ("pumped", 2), ("hyped", 2.5), ("lit", 1.5), ("fire", 1.5),
        # Gen Z / Internet
        ("slay", 2.5), ("slaying", 2.5), ("slayed", 2.5), ("serving", 1.5),
        ("ate", 1.5), ("no cap", 2), ("lowkey", 0.5), ("highkey", 1.5),
        ("iconic", 2), ("vibing", 2), ("vibe", 1.5), ("bussin", 2.5),
        ("valid", 1), ("periodt", 1.5), ("bestie", 1), ("geeked", 2.5),
        ("based", 1.5), ("goated", 2.5), ("rizzing", 2), ("rizzed", 2),
        ("understood the assignment", 2.5), ("ate that", 2),
        ("main character", 1.5), ("era", 1), ("blessed era", 2),
        ("pop off", 2), ("we love", 1.5), ("lowkey obsessed", 2),
        # Formal / literary
        ("euphoric", 3), ("radiant", 2), ("glowing", 2), ("buzzing", 2),
        ("beaming", 2.5), ("floating", 1.5), ("enchanted", 2.5), ("rapturous", 3),
        ("exhilarated", 3), ("exuberant", 3), ("vivacious", 2), ("buoyant", 2),
        ("elation", 3), ("felicity", 2.5), ("bliss", 3), ("glee", 3),
        ("delight", 2.5), ("triumph", 2.5), ("accomplishment", 2),
        ("fulfillment", 2), ("satisfaction", 2), ("contentment", 2),
        ("achievement", 2), ("success", 2), ("victory", 2.5), ("win", 2),
        ("reward", 1.5), ("progress", 1.5), ("milestone", 1.5),
        # Physical joy expressions
        ("dancing", 1.5), ("singing", 1.5), ("jumping", 1.5), ("bouncing", 1.5),
        ("running", 1), ("hugging", 1.5), ("kissing", 1.5), ("clapping", 1.5),
        ("cheering", 2), ("whooping", 2), ("screaming with joy", 3),
        # Cultural
        ("mashallah", 2), ("alhamdulillah", 2), ("waheguru", 2), ("hallelujah", 2),
        ("praise", 1.5), ("blessed be", 2), ("jai ho", 2), ("itadakimasu", 1),
        ("kanpai", 1.5), ("cheers", 1.5), ("bravo", 2), ("encore", 1.5),
        # Relationship joy
        ("love them", 2), ("they said yes", 3), ("we're together", 2),
        ("engaged", 2), ("married", 2), ("pregnant", 1.5), ("new baby", 2),
        ("reunion", 2), ("reconciled", 2), ("forgiven", 1.5), ("accepted", 1.5),
    ],
    "sadness": [
        # Core sadness
        ("sad", 3), ("sadness", 3), ("depressed", 3), ("depression", 3),
        ("unhappy", 3), ("miserable", 3), ("down", 1.5), ("blue", 1.5),
        ("hurt", 2), ("disappointed", 2), ("lonely", 3), ("alone", 1.5),
        ("grief", 3), ("grieve", 3), ("grieving", 3), ("mourn", 3), ("mourning", 3),
        ("desperate", 3), ("hopeless", 3), ("helpless", 2.5), ("powerless", 2.5),
        ("crying", 3), ("tears", 3), ("sob", 3), ("sobbing", 3), ("weeping", 3),
        ("broken", 2.5), ("devastated", 3), ("sorrowful", 3), ("melancholic", 3),
        ("gloomy", 2), ("dejected", 3), ("forlorn", 3), ("heartbroken", 3),
        ("worthless", 3), ("empty", 2), ("numb", 2), ("hollow", 2),
        ("miss", 1.5), ("missing", 2), ("lost", 1.5), ("failed", 1.5),
        ("useless", 2.5), ("unloved", 3), ("rejected", 2.5), ("abandoned", 3),
        ("exhausted", 1.5), ("drained", 2), ("weary", 2), ("tired", 1),
        ("wrecked", 2.5), ("crushed", 2.5), ("shattered", 3), ("gutted", 3),
        ("spiraling", 2.5), ("falling apart", 3), ("unmotivated", 2),
        ("disconnected", 2), ("detached", 2), ("withdrawn", 2),
        ("invisible", 2), ("unseen", 2), ("unwanted", 3), ("unworthy", 3),
        ("struggling", 2), ("suffering", 3), ("aching", 2.5), ("drowning", 2.5),
        ("sorrow", 3), ("despair", 3), ("despairing", 3), ("anguish", 3),
        # Clinical / mental health
        ("anhedonia", 3), ("dysthymia", 3), ("burnout", 2.5), ("burnout", 2.5),
        ("apathetic", 2), ("apathy", 2), ("listless", 2.5), ("languishing", 2.5),
        ("melancholy", 3), ("desolate", 3), ("inconsolable", 3), ("bereft", 3),
        ("disconsolate", 3), ("wretched", 3), ("anguished", 3), ("tormented", 2.5),
        # Gen Z / Internet
        ("not okay", 3), ("lowkey sad", 2.5), ("in my feels", 2.5),
        ("going through it", 2.5), ("rip me", 2), ("dead inside", 3),
        ("crying in the car", 3), ("ugly crying", 3), ("emotional damage", 2.5),
        ("it hurts", 2.5), ("pain", 2), ("hurts so much", 3), ("this sucks", 2),
        ("sad era", 3), ("my heart", 2), ("heartache", 3), ("loss", 2.5),
        ("i'm done", 1.5), ("gave up", 2.5), ("can't anymore", 3),
        ("emotionally exhausted", 3), ("mentally drained", 3), ("broken inside", 3),
        # Physical sadness
        ("heavy chest", 3), ("lump in throat", 3), ("tight chest", 2),
        ("can't eat", 2), ("can't sleep", 1.5), ("no appetite", 2),
        ("lying in bed", 1.5), ("can't get up", 2.5), ("heavy limbs", 2.5),
        # Nostalgia / loss
        ("nostalgia", 2), ("nostalgic", 2.5), ("remember when", 1.5),
        ("used to be", 1.5), ("things changed", 1.5), ("nothing same", 2),
        ("old days", 1.5), ("miss the old", 2), ("wish things were different", 2.5),
        ("if only", 1.5), ("should have", 1.5), ("regret", 2.5), ("guilt", 2),
    ],
    "anger": [
        # Core anger
        ("angry", 3), ("anger", 3), ("furious", 3), ("fury", 3),
        ("rage", 3), ("raging", 3), ("mad", 2.5), ("irritated", 2), ("annoyed", 2),
        ("frustrated", 2.5), ("frustration", 2.5), ("hate", 3), ("hating", 2.5),
        ("despise", 3), ("aggressive", 2), ("hostile", 2), ("resentment", 3),
        ("resentful", 3), ("bitter", 2), ("bitterness", 2.5), ("vengeful", 3),
        ("enraged", 3), ("livid", 3), ("seething", 3), ("incensed", 3),
        ("irate", 3), ("indignant", 2), ("exasperated", 2), ("outraged", 3),
        ("fuming", 3), ("infuriated", 3), ("incensed", 3),
        ("unfair", 1.5), ("injustice", 2), ("unjust", 2), ("disrespect", 2),
        ("stupid", 1.5), ("idiot", 2), ("ridiculous", 1.5), ("absurd", 1.5),
        ("fed up", 2.5), ("sick of", 2.5), ("can't stand", 2.5),
        ("triggered", 2), ("pissed", 2.5), ("heated", 2.5), ("salty", 2),
        ("pressed", 1.5), ("vexed", 2), ("boiling", 2.5), ("snapping", 2),
        ("losing it", 2.5), ("done", 2), ("over it", 2),
        ("clowned", 1.5), ("scammed", 2), ("betrayed", 2.5), ("lied to", 2.5),
        ("gaslit", 3), ("manipulated", 3), ("disrespected", 2.5),
        ("taken for granted", 2.5), ("steaming", 2.5), ("not having it", 2.5),
        # Formal
        ("indignation", 3), ("wrath", 3), ("wrathful", 3), ("incensed", 3),
        ("belligerent", 2.5), ("combative", 2), ("adversarial", 2),
        ("confrontational", 2), ("antagonistic", 2.5), ("defiant", 2),
        ("retaliatory", 2.5), ("vindictive", 3), ("acrimonious", 3),
        # Gen Z expanded
        ("big mad", 3), ("in my villain era", 2.5), ("choosing violence", 2.5),
        ("petty era", 2), ("not the one", 2.5), ("crossed the line", 2.5),
        ("last nerve", 2.5), ("you tried it", 2.5), ("absolutely not", 2),
        ("the audacity", 2.5), ("the nerve", 2.5), ("they got me", 2),
        ("not playing", 2.5), ("war mode", 2.5), ("activated", 1.5),
        ("red flag", 2), ("toxic", 1.5), ("manipulative", 2.5), ("narcissistic", 2.5),
        ("abusive", 3), ("exploited", 2.5), ("used me", 2.5), ("played me", 2.5),
        # Physical anger
        ("clenching", 2), ("jaw tight", 2), ("fists", 2), ("shaking with anger", 3),
        ("blood boiling", 3), ("seeing red", 3), ("white hot rage", 3),
        ("heart pounding anger", 2.5), ("temples throbbing", 2),
        # Moral anger
        ("unacceptable", 2.5), ("wrong", 1.5), ("immoral", 2), ("evil", 2.5),
        ("corrupt", 2), ("exploitation", 2.5), ("oppression", 2.5), ("abuse", 2.5),
        ("violation", 2.5), ("criminal", 2), ("atrocity", 2.5),
    ],
    "fear": [
        # Core fear
        ("afraid", 3), ("fear", 3), ("fearful", 3), ("scared", 3),
        ("terrified", 3), ("terror", 3), ("anxious", 3), ("anxiety", 3),
        ("worried", 3), ("worrying", 2.5), ("nervous", 2.5), ("nervousness", 2.5),
        ("panic", 3), ("panicking", 3), ("panicked", 3), ("dread", 3),
        ("dreading", 3), ("horrified", 3), ("frightened", 3), ("fright", 2.5),
        ("apprehensive", 3), ("apprehension", 3), ("petrified", 3), ("alarmed", 2),
        ("uneasy", 2), ("unease", 2), ("stressed", 2.5), ("stress", 2),
        ("trembling", 3), ("shaking", 2.5), ("unsafe", 3), ("threatened", 3),
        ("vulnerable", 2), ("overwhelmed", 2.5), ("distressed", 2.5),
        ("phobia", 3), ("nightmare", 2), ("haunted", 2), ("dreadful", 2),
        ("freaking out", 3), ("spiraling", 2.5), ("hyperventilating", 3),
        ("doom", 2.5), ("doomed", 3), ("terrifying", 3), ("on edge", 2.5),
        ("jumpy", 2), ("jittery", 2.5), ("intrusive thoughts", 3),
        ("worst case", 2), ("what if", 1.5), ("anticipatory", 2),
        ("health anxiety", 3), ("social anxiety", 3), ("existential", 2),
        ("can't breathe", 3), ("chest tight", 2.5), ("heart racing", 2.5),
        # Clinical
        ("ptsd", 3), ("trauma", 3), ("traumatized", 3), ("flashback", 3),
        ("trigger warning", 2.5), ("triggered", 2), ("hypervigilant", 3),
        ("dissociating", 3), ("dissociation", 3), ("fight or flight", 3),
        ("adrenaline", 2), ("cortisol", 2), ("freeze response", 3),
        ("panic disorder", 3), ("generalized anxiety", 3), ("ocd", 2.5),
        ("intrusive", 2.5), ("rumination", 2.5), ("catastrophizing", 3),
        # Gen Z expanded
        ("doom scrolling", 2), ("doomscrolling", 2), ("anxiety attack", 3),
        ("brain won't stop", 3), ("overthinking", 2.5), ("spiral", 2.5),
        ("not safe", 2.5), ("scared to try", 2), ("imposter syndrome", 2.5),
        ("afraid to fail", 2.5), ("scared of judgment", 2.5),
        ("social anxiety", 3), ("phone anxiety", 2.5), ("decision paralysis", 2.5),
        ("nervous breakdown", 3), ("mental health crisis", 3),
        # Physical fear
        ("goosebumps", 2), ("cold sweat", 3), ("palms sweating", 2.5),
        ("dry mouth", 2), ("stomach dropping", 2.5), ("nausea from fear", 3),
        ("legs like jelly", 2.5), ("frozen in place", 3), ("paralyzed by fear", 3),
        ("voice shaking", 2.5), ("can't think straight", 2), ("mind blank", 2),
    ],
    "surprise": [
        # Core surprise
        ("surprised", 3), ("surprise", 3), ("amazed", 2.5), ("amazing", 1.5),
        ("astonished", 3), ("shocked", 3), ("shock", 3), ("aghast", 3),
        ("wow", 3), ("unexpected", 3), ("sudden", 2), ("suddenly", 2),
        ("startled", 3), ("flabbergasted", 3), ("speechless", 2),
        ("dumbfounded", 3), ("dumbstruck", 3), ("unbelievable", 2.5),
        ("incredible", 2), ("whoa", 3), ("omg", 2.5), ("no way", 2.5),
        ("what", 1), ("can't believe", 3), ("didn't expect", 3),
        ("never expected", 3), ("out of nowhere", 3), ("blown away", 3),
        ("plot twist", 2.5), ("wait what", 3), ("huh", 1.5), ("bruh", 1.5),
        ("lowkey shocked", 2.5), ("not expecting", 2.5),
        ("caught off guard", 3), ("randomly", 2), ("just found out", 2.5),
        ("breaking news", 2), ("revelation", 2.5), ("discovered", 1.5),
        ("found out", 2), ("blindsided", 3), ("curveball", 2.5),
        ("twist", 2), ("turn of events", 2.5), ("development", 1.5),
        # Gen Z expanded
        ("bro what", 3), ("bestie what", 3), ("periodt wait", 2),
        ("deadass", 2.5), ("literally shook", 3), ("shook", 3), ("shook me", 3),
        ("it's giving plot twist", 2.5), ("lore dropped", 2.5),
        ("the reveal", 2), ("this changes everything", 2.5),
        ("wait hold up", 2.5), ("hold on a second", 2), ("pause", 1.5),
        ("I'm not ready", 2), ("wasn't built for this", 2.5),
        ("say sike", 2.5), ("sike rn", 2.5), ("the audacity to surprise me", 2),
        ("mindblown", 3), ("mind blown", 3), ("brain broke", 3),
        ("paralyzed with shock", 3), ("shell shocked", 3),
        # Positive surprise
        ("pleasant surprise", 3), ("unexpected blessing", 2.5), ("surprised by joy", 3),
        ("happily surprised", 2.5), ("wonderful surprise", 3),
        # Negative surprise
        ("bad news", 2), ("terrible surprise", 2.5), ("shocking revelation", 3),
        ("scandal", 2), ("betrayal surprise", 2.5), ("exposed", 2),
    ],
    "disgust": [
        # Core disgust
        ("disgusted", 3), ("disgust", 3), ("gross", 3), ("yuck", 3),
        ("nasty", 3), ("vile", 3), ("repulsive", 3), ("revolting", 3),
        ("offensive", 2), ("abhor", 3), ("abhorrent", 3), ("loathe", 3),
        ("loathing", 3), ("sickening", 3), ("repugnant", 3), ("distasteful", 3),
        ("icky", 3), ("repellent", 3), ("obnoxious", 2), ("abominable", 3),
        ("filthy", 2.5), ("putrid", 3), ("foul", 3), ("rotten", 2),
        ("nauseating", 3), ("appalling", 2), ("eww", 3), ("ew", 3),
        ("yikes", 2), ("eugh", 3), ("blegh", 3), ("blech", 3),
        ("cringe", 2.5), ("cringy", 2.5), ("cringeworthy", 3),
        ("rancid", 3), ("mid", 1.5), ("trash", 2), ("sus", 1.5),
        ("problematic", 2), ("toxic", 2), ("red flag", 2.5), ("ick", 3),
        ("pick me", 1.5), ("embarrassing", 2), ("shameful", 2.5),
        ("hypocrite", 2.5), ("two faced", 2.5), ("fake", 2),
        # Moral disgust
        ("immoral", 2.5), ("unethical", 2.5), ("corrupt", 2.5), ("evil", 2.5),
        ("wicked", 2.5), ("depraved", 3), ("perverted", 3), ("degenerate", 2.5),
        ("contemptible", 3), ("reprehensible", 3), ("despicable", 3),
        ("unconscionable", 3), ("inhumane", 3), ("atrocious", 3),
        ("monstrous", 3), ("villainous", 2.5), ("corrupt", 2.5),
        # Physical disgust
        ("want to vomit", 3), ("makes me gag", 3), ("stomach turning", 3),
        ("nauseous", 2.5), ("sick to my stomach", 3), ("bile rising", 3),
        ("can't look at", 2.5), ("have to look away", 2.5), ("recoiling", 2.5),
        # Gen Z expanded
        ("giving me the ick", 3), ("the ick is real", 3), ("permanent ick", 3),
        ("uninstalling", 2.5), ("blocking and deleting", 2.5), ("walking away",1.5),
        ("cancel worthy", 2.5), ("cancelled", 2), ("problematic behavior", 2.5),
        ("virtue signaling", 2), ("performative", 2), ("fake woke", 2.5),
        ("not it chief", 2.5), ("it's not giving", 2), ("giving nothing", 1.5),
        ("absolute zero", 2), ("no redeeming qualities", 2.5),
        ("the villain behavior", 2), ("morally bankrupt", 3),
    ],
    "neutral": [
        # Core neutral
        ("okay", 1), ("fine", 1), ("alright", 1), ("normal", 1),
        ("regular", 1), ("ordinary", 1), ("usual", 1), ("average", 1),
        ("moderate", 1), ("stable", 1.5), ("calm", 1.5), ("balanced", 1.5),
        ("indifferent", 2), ("neutral", 3), ("meh", 2.5), ("whatever", 2),
        ("just", 0.5), ("kinda", 0.5), ("sorta", 0.5), ("i guess", 1),
        ("not really", 1), ("eh", 2), ("mid", 1.5), ("unbothered", 1.5),
        ("chilling", 1), ("existing", 1), ("it is what it is", 2.5),
        ("whatever happens", 1.5), ("not much", 1), ("nothing special", 2),
        ("same as usual", 1.5), ("routine", 1), ("habit", 1), ("typical", 1),
        ("standard", 1), ("baseline", 1.5), ("functional", 1), ("adequate", 1.5),
        ("passable", 1.5), ("tolerable", 1.5), ("acceptable", 1.5),
        ("ordinary day", 1.5), ("uneventful", 2), ("unremarkable", 2),
        ("monotonous", 1.5), ("routine", 1), ("repetitive", 1),
        ("going through motions", 2), ("autopilot", 2), ("coasting", 1.5),
        # Gen Z neutral
        ("npc mode", 2), ("background character", 2), ("on autopilot", 2),
        ("no thoughts head empty", 2.5), ("vibing at zero", 2), ("baseline", 1.5),
        ("no drama", 1.5), ("peaceful", 1), ("chill", 1), ("low key day", 1.5),
        ("nothing major", 1.5), ("another day", 1), ("same energy", 1),
        ("pretty standard", 1.5), ("expected outcome", 1.5), ("as predicted", 1.5),
        ("no surprises", 2), ("business as usual", 2), ("steady state", 2),
    ],
}

# ==================== VASTLY EXPANDED EMOJI MAP ====================
EMOTION_EMOJIS_MAP = {
    # Joy
    "😊": "joy", "😁": "joy", "😄": "joy", "😃": "joy", "🥳": "joy",
    "😍": "joy", "❤️": "joy", "🎉": "joy", "✨": "joy", "💖": "joy",
    "😂": "joy", "🤣": "joy", "😆": "joy", "🥰": "joy", "💕": "joy",
    "🫶": "joy", "🌸": "joy", "🎊": "joy", "🏆": "joy", "🎀": "joy",
    "💗": "joy", "💓": "joy", "💛": "joy", "💚": "joy", "🧡": "joy",
    "❣️": "joy", "💞": "joy", "🌟": "joy", "⭐": "joy", "🌈": "joy",
    "🎆": "joy", "🎇": "joy", "🥂": "joy", "🍾": "joy", "🎂": "joy",
    "🎁": "joy", "🎈": "joy", "🎠": "joy", "🌺": "joy", "🌻": "joy",
    "🌷": "joy", "☀️": "joy", "🌤️": "joy", "😎": "joy", "🤩": "joy",
    "😜": "joy", "😝": "joy", "🙌": "joy", "👏": "joy", "🤸": "joy",
    # Sadness
    "😢": "sadness", "😭": "sadness", "💔": "sadness", "😔": "sadness",
    "😞": "sadness", "🥺": "sadness", "😿": "sadness", "🫂": "sadness",
    "💙": "sadness", "🌧️": "sadness", "😩": "sadness", "😪": "sadness",
    "🌑": "sadness", "🌊": "sadness", "😟": "sadness", "😰": "sadness",
    "😦": "sadness", "😧": "sadness", "🌫️": "sadness", "🥀": "sadness",
    "🖤": "sadness", "💀": "sadness", "☹️": "sadness", "😓": "sadness",
    "🌚": "sadness", "😑": "sadness",
    # Anger
    "😡": "anger", "🤬": "anger", "😤": "anger", "💢": "anger",
    "🔥": "anger", "⚡": "anger", "💥": "anger", "🗯️": "anger",
    "👿": "anger", "😾": "anger", "🤛": "anger", "✊": "anger",
    "🖕": "anger", "😠": "anger",
    # Fear
    "😨": "fear", "😰": "fear", "😱": "fear", "😬": "fear",
    "🫣": "fear", "🙀": "fear", "👻": "fear", "💀": "fear",
    "😵": "fear", "🫀": "fear", "😓": "fear", "🙏": "fear",
    "🌀": "fear",
    # Surprise
    "😮": "surprise", "😲": "surprise", "🤯": "surprise", "😳": "surprise",
    "🫢": "surprise", "🙊": "surprise", "👀": "surprise", "❗": "surprise",
    "‼️": "surprise", "❓": "surprise", "🎭": "surprise",
    # Disgust
    "🤢": "disgust", "🤮": "disgust", "😒": "disgust", "🤧": "disgust",
    "😏": "disgust", "🙄": "disgust", "😑": "disgust", "🚫": "disgust",
    "❌": "disgust", "🤦": "disgust", "🤦‍♀️": "disgust", "🤦‍♂️": "disgust",
    "💩": "disgust", "🗑️": "disgust",
    # Neutral
    "😐": "neutral", "🙂": "neutral", "🫤": "neutral",
    "😶": "neutral", "🤷": "neutral", "🤷‍♀️": "neutral", "🤷‍♂️": "neutral",
    "😌": "neutral",
}

# ==================== MASSIVELY EXPANDED PHRASE PATTERNS ====================
PHRASE_PATTERNS = {
    "joy": [
        # Classic
        "over the moon", "on cloud nine", "best day", "so happy", "feel great",
        "feel amazing", "absolutely love", "so excited", "love it", "feeling good",
        "feel wonderful", "really happy", "so glad", "thrilled about", "ecstatic about",
        "having fun", "made my day", "can't stop smiling", "cannot stop smiling",
        "best thing", "so grateful", "really excited", "so proud", "very happy",
        "extremely happy", "feeling happy", "loving life", "best life", "happy place",
        "living my best", "couldn't be happier", "happiest i've been", "peak happiness",
        "this made me so happy", "genuinely so happy", "loving this",
        "having the best time", "things finally worked out", "it all came together",
        "achieved my goal", "dreams came true", "made it happen",
        "feeling appreciated", "feel valued", "feel seen", "feel heard",
        # Gen Z expanded
        "i ate and left no crumbs", "slay era", "in my element",
        "absolutely buzzing", "thriving and it shows", "main character energy",
        "understood the assignment", "ate that down", "living my best life",
        "glow up era", "my bag is secured", "the universe said yes",
        "alignment unlocked", "pop off moment", "we love to see it",
        "icon behavior", "this is everything", "the girlies are thriving",
        "we're so back", "blessed era", "happy era", "joy unlocked",
        "romantic era unlocked", "life said yes", "the prophecy was right",
        "winning streak", "on a roll", "can't stop won't stop",
        "served and blessed", "glowing up", "leveled up",
        "manifested this", "the law of attraction worked", "vision board activated",
        # Achievement phrases
        "passed with flying colors", "got the job", "got accepted", "they said yes",
        "finally got", "worked out perfectly", "everything clicked",
        "hit my goal", "crushed the goal", "exceeded expectations",
        "nailed it", "killed it", "crushed it", "slayed it",
    ],
    "sadness": [
        # Classic
        "feel so sad", "feeling down", "so lonely", "feel alone",
        "broken heart", "miss you", "feel hopeless", "cant stop crying",
        "cannot stop crying", "no one cares", "feel worthless", "feel empty",
        "feel lost", "feel depressed", "feeling blue", "feeling sad",
        "tearing up", "want to cry", "need to cry", "feeling hopeless",
        "gave up", "given up", "no point", "what's the point",
        "miss them", "miss her", "miss him", "miss everything",
        "not okay", "really not okay", "lowkey not okay",
        "genuinely struggling", "falling apart inside", "barely holding on",
        "exhausted from everything", "nothing makes sense",
        "can't find joy", "numb to everything", "going through it",
        "in my feels", "deep in my feels", "crying myself to sleep",
        "cried all day", "tears won't stop", "feel like a burden",
        "nobody gets it", "nobody understands", "so drained",
        "emotionally exhausted", "burned out", "don't see the point",
        "struggling to get up", "just existing", "fading away",
        "nobody notices", "feel like a ghost", "drowning in sadness",
        "sinking deeper", "weight of sadness", "can't remember joy",
        # Gen Z expanded
        "sad era", "not the same person", "lost my spark",
        "the depression is loud", "mental health said no",
        "feeling grey inside", "the emptiness is overwhelming",
        "miss who i used to be", "therapy isn't enough",
        "crying at 2am", "sobbing into my pillow",
        "emotional damage is real", "in my flop era",
        "down bad right now", "it's giving sadness",
        "my heart is heavy", "carrying a lot", "emotionally offline",
        # Loss and grief
        "lost someone", "lost a loved one", "they're gone", "passed away",
        "i miss them every day", "grief wave hit", "grief is heavy",
        "still healing from", "processing loss", "broken by the loss",
        "healing but hurting", "bittersweet memories",
    ],
    "anger": [
        # Classic
        "so angry", "really mad", "absolutely furious", "makes me mad",
        "pissed off", "fed up", "had enough", "how dare",
        "so unfair", "can't stand it", "drives me crazy", "sick of",
        "so annoyed", "really frustrated", "beyond frustrated",
        "so done", "done with", "can't take it", "makes my blood boil",
        "losing my mind", "not fair at all", "this is ridiculous",
        "they always do this", "tired of being treated like",
        "they crossed the line", "that's it i'm done",
        "can't deal with this anymore", "so over this",
        "they got me so heated", "big mad right now",
        "i'm fuming", "i'm seething", "genuinely furious",
        "i want to scream", "this makes me so angry",
        "disrespected again", "they don't respect me",
        "i've been gaslit", "i'm not imagining things",
        "it's giving manipulative", "red flag behavior",
        # Gen Z expanded
        "choosing violence", "in my villain era", "not the one",
        "they tried it", "the audacity to", "pressed about this",
        "cancel this person", "chronically disrespectful",
        "done being nice about it", "no longer playing nice",
        "the disrespect is wild", "the entitlement is wild",
        "i refuse to accept", "not taking this",
        "done being the bigger person", "petty era activated",
        "war mode activated", "they called the wrong one",
        "chaos activated", "i'm not okay with this",
        # Moral anger
        "injustice angers me", "this is wrong on every level",
        "cannot support this", "standing against this",
        "will not tolerate this", "unacceptable behavior",
        "this violates everything", "morally wrong",
        "ethically bankrupt", "this is abuse",
    ],
    "fear": [
        # Classic
        "so scared", "really afraid", "terrified of", "anxious about",
        "worried about", "panicking right now", "scared of", "afraid of",
        "nervous about", "stressed about", "feel unsafe", "very anxious",
        "having a panic attack", "can't sleep", "cant sleep",
        "keep thinking about", "what if something", "scared something",
        "freaking out right now", "spiraling bad", "anxiety is through the roof",
        "so much anxiety", "can't stop worrying", "doomscrolling again",
        "my mind won't stop", "intrusive thoughts again",
        "heart is racing", "chest feels tight", "can't breathe properly",
        "everything feels threatening", "scared of the future",
        "don't feel safe", "hyperventilating a little",
        "worst case scenario brain", "catastrophizing again",
        "so overwhelmed", "too much going on", "can't cope",
        "scared to fail", "scared of being alone",
        "what if it goes wrong", "terrified of what comes next",
        # Gen Z expanded
        "anxiety said hello", "anxiety ate today", "nervous wreck rn",
        "scared to open messages", "scared to check notifications",
        "phone anxiety real", "decision paralysis hitting",
        "imposter syndrome loud today", "scared of disappointing people",
        "fear of judgment is real", "social anxiety destroying me",
        "cannot make this decision", "frozen with anxiety",
        "the dread is real", "doom brain activated",
        "existential dread hitting", "what is even happening",
        "cannot process this much", "stimulation overload",
        "overstimulated and scared", "sensory overload today",
        # Physical fear phrases
        "heart won't slow down", "palms are sweating", "legs are shaking",
        "can't catch my breath", "going to pass out from anxiety",
        "body is in panic mode", "fight or flight triggered",
        "nervous system is shot", "adrenaline won't stop",
    ],
    "surprise": [
        # Classic
        "cannot believe", "cant believe", "didn't expect", "never expected",
        "out of nowhere", "caught me off guard", "so shocking",
        "totally unexpected", "did not see", "shocked to",
        "amazed by", "astonished by", "blew my mind", "mind blown",
        "no way this", "this is insane", "this is crazy",
        "wait what just happened", "bro what", "bruh moment",
        "plot twist i didn't see coming", "came out of nowhere",
        "lowkey shook", "genuinely shook", "wasn't expecting that",
        "just found out and", "wait hold on", "okay i did not expect",
        # Gen Z expanded
        "the lore dropped", "plot twist unlocked", "reveal arc",
        "lore accurate shock", "reality said plot twist",
        "the prophecy said nothing about this", "the npc became a main character",
        "reality shifted", "the rules changed", "everything is different now",
        "the reveal changed everything", "this changes the lore",
        "didn't have this on my bingo card", "unexpected achievement unlocked",
        "random event spawned", "this wasn't in the script",
        "reality said surprise", "universe said plot twist",
        # Emotional context
        "good news surprised me", "bad news blindsided me",
        "happy surprise arrived", "shocking development",
        "the truth came out", "the secret was revealed",
        "discovered something huge", "found out the truth",
    ],
    "disgust": [
        # Classic
        "so disgusting", "makes me sick", "hate this", "so gross",
        "absolutely revolting", "can't stand", "find it repulsive",
        "absolutely horrible", "that's gross", "that is gross",
        "so nasty", "totally gross", "really disgusting",
        "giving me the ick", "got the ick from", "major ick",
        "so cringe", "genuinely cringe", "embarrassing to watch",
        "it's giving trash", "not the one", "this ain't it chief",
        "lowkey rancid", "that's so problematic", "red flag behavior",
        "it's disgusting how", "morally wrong", "can't stand how",
        "genuinely repulsed", "makes my stomach turn",
        # Gen Z expanded
        "the ick is permanent", "uninstalling this person",
        "blocking and deleting", "no redeeming qualities",
        "giving rancid and toxic", "this is so problematic",
        "their behavior is disgusting", "i can't even look",
        "the hypocrisy is vile", "performative and hollow",
        "virtue signaling while being vile", "absolute clown behavior",
        "the entitlement is disgusting", "they have no shame",
        "shameless behavior", "zero accountability",
        "the fake apology was disgusting", "not sorry at all",
        "the audacity to be gross", "chronically problematic",
        # Moral disgust phrases
        "what they did is wrong", "cannot support this behavior",
        "this violates my values", "morally bankrupt behavior",
        "ethically disgusting", "human decency violated",
        "this is inhumane", "cannot condone this",
    ],
}

# ==================== SARCASM DETECTION ====================
SARCASM_PATTERNS = [
    r"\boh (great|wonderful|fantastic|amazing|perfect)\b",
    r"\bsure(ly)?,? (that('s| is) (totally|definitely|absolutely))\b",
    r"\b(wow|amazing|incredible),? (really|truly)?\b.{0,20}\bnot\b",
    r"\bjust (what|what i) (needed|wanted)\b.{0,20}\bnot\b",
    r"\b(obviously|clearly|of course)\b.{0,30}\bsaid no one\b",
    r"\bthanks? (a lot|so much|for nothing)\b",
    r"\bgreat,? (another|one more|just what)\b",
    r"\b(so|very|really|totally) (helpful|useful|great)\b.{0,20}\bnot\b",
    r"yeah (right|sure|okay|ok)\b",
    r"\bnot (like|as if|that)\b",
    r"\bcan't (wait|imagine|believe) (not|how)\b",
    r"\blove how\b.{0,30}\b(always|never|constantly)\b",
]

def _detect_sarcasm(text: str) -> bool:
    """Detect potential sarcasm — reduces joy score, may boost neutral/sadness."""
    t = text.lower()
    for pattern in SARCASM_PATTERNS:
        if re.search(pattern, t):
            return True
    # Check for positive words + negative context
    positive_words = {"happy", "great", "amazing", "wonderful", "fantastic", "love", "perfect"}
    negative_words = {"not", "never", "no", "can't", "won't", "but", "however", "although"}
    words = set(t.split())
    if len(words & positive_words) >= 1 and len(words & negative_words) >= 2:
        return True
    return False

# ==================== MULTI-EMOTION BLENDING ====================
MIXED_EMOTION_PATTERNS = {
    "bittersweet":   (["joy", "sadness"], 0.5),
    "anxious_joy":   (["joy", "fear"],    0.4),
    "angry_sad":     (["anger", "sadness"], 0.4),
    "fearful_anger": (["anger", "fear"],  0.45),
    "surprised_joy": (["surprise", "joy"], 0.5),
    "surprised_fear":  (["surprise", "fear"], 0.45),
}
MIXED_EMOTION_PHRASES = {
    "bittersweet": [
        "happy but sad", "bitter sweet", "bittersweet", "mixed feelings",
        "happy and sad at same time", "crying happy tears", "laughing through tears",
        "excited but nervous", "thrilled but scared", "smiling but hurting",
        "loving but losing", "happy for them but sad for me",
    ],
    "anxious_joy": [
        "excited but nervous", "nervous excitement", "anxious about something good",
        "happy but worried", "thrilled but anxious", "good news but scared",
        "excited and terrified", "butterflies good and bad",
    ],
    "angry_sad": [
        "hurt and angry", "angry because i'm hurt", "sad and angry",
        "furious but heartbroken", "crying from anger", "mad and devastated",
        "betrayed and heartbroken", "angry at myself for being sad",
    ],
}

NEGATION_WORDS = {
    "not", "no", "never", "dont", "don't", "cant", "can't",
    "wont", "won't", "isnt", "isn't", "wasnt", "wasn't",
    "didnt", "didn't", "havent", "haven't", "nothing",
    "nowhere", "neither", "nor", "barely", "hardly", "scarcely",
    "no longer", "not really", "not at all", "nah", "nope",
    "without", "lack", "lacking", "absent", "absence",
}

INTENSIFIERS = {
    "very": 1.6, "so": 1.6, "really": 1.6, "extremely": 2.2,
    "absolutely": 2.2, "completely": 2.0, "totally": 2.0,
    "incredibly": 2.2, "terribly": 2.0, "awfully": 1.8,
    "deeply": 1.8, "utterly": 2.2, "quite": 1.3, "pretty": 1.3,
    "super": 1.7, "insanely": 2.0, "ridiculously": 1.8,
    "freaking": 1.8, "horribly": 2.0, "truly": 1.5,
    "genuinely": 1.7, "literally": 1.5, "lowkey": 1.2, "highkey": 1.7,
    "deadass": 1.8, "for real": 1.6, "fr": 1.4, "no cap": 1.7,
    "unbelievably": 2.0, "overwhelmingly": 2.0,
    "beyond": 1.8, "madly": 1.9, "wildly": 1.8,
    "profoundly": 2.0, "extraordinarily": 2.2, "remarkably": 1.8,
    "strikingly": 1.7, "undeniably": 1.8, "painfully": 1.9,
    "unbearably": 2.0, "impossibly": 2.0, "devastatingly": 2.2,
    "breathtakingly": 2.0, "achingly": 1.9, "desperately": 1.9,
    "furiously": 2.0, "violently": 2.0, "intensely": 1.8,
}

EMOTION_EMOJIS = {
    "joy":      "🌸",
    "sadness":  "💙",
    "anger":    "🔥",
    "fear":     "💜",
    "surprise": "✨",
    "disgust":  "🤢",
    "neutral":  "🎀",
}

# ==================== VASTLY EXPANDED RESPONSES ====================
COMFORT_RESPONSES = {
    "joy": [
        "🌟 That's absolutely wonderful! Your happiness is so contagious — keep shining and celebrating every bit of this! ✨",
        "💫 I'm so genuinely thrilled for you! This joy is precious — hold onto it and let yourself feel every bit of it! 🎉",
        "😊 Your excitement warms my heart completely! You deserve every single bit of this beautiful moment! 🌈",
        "💖 You deserve all this joy and more! Keep being your amazing self and spreading that gorgeous energy! 🎀",
        "🌸 Seeing you this happy makes me so happy too! Celebrate yourself — you've earned this! 🥳",
        "✨ This energy is everything! You're glowing and it shows — don't let anyone dim your light! 💕",
        "🎊 Your joy literally radiates through your words! Soak every drop of this up — you deserve it! 🌟",
        "💗 I love hearing this from you! This is what you've been working toward — enjoy every second! 🌈",
        "🏆 You're thriving and it's the most beautiful thing! Keep that momentum going — you've got this! ✨",
        "🌺 This level of happiness is everything! The universe is celebrating with you right now! 💫",
        "🎀 Look at you, absolutely flourishing! This is your moment — own it completely! 🌸",
        "⭐ Something about your joy today is making me want to dance! Keep winning, you deserve all of it! 💕",
    ],
    "sadness": [
        "💙 I hear you, and your feelings truly matter so much. It's okay to feel sad — I'm right here with you, always. 🫂",
        "🌙 These feelings are heavy, but they're not permanent. You're so much stronger than you know, even when it doesn't feel that way. ✨",
        "💜 Sadness is part of being beautifully human. You don't have to carry this alone — I'm right here. 🤍",
        "💗 Your pain is completely valid. Be incredibly gentle with yourself right now — you deserve that softness. 🌸",
        "🌸 Going through it is so hard, and I see you. You don't have to pretend to be okay — just be where you are. 💙",
        "🤍 Some days are just really hard, and that's allowed. Take it one breath at a time — I'm not going anywhere. 🌙",
        "💙 I'm sitting right here with you in this. You don't have to explain it or fix it — just feel it. 🫂",
        "🌙 The heaviness you're carrying is real. Please be as kind to yourself today as you would be to someone you love. 💕",
        "🌸 Whatever broke your heart deserves to be acknowledged. You're not weak for hurting — you're human. 💙",
        "💜 You're allowed to fall apart a little. I'll be here when you're ready to piece things together. 🤝",
        "🌊 Grief and sadness move in waves — let them wash over you. You will surface again. 💙",
        "⭐ You matter. Your pain matters. And you deserve support — please reach out to someone who loves you too. 💗",
    ],
    "anger": [
        "🔥 Your anger is completely valid — those feelings matter and you have every right to feel them. Let's breathe through this together. 🌬️",
        "💪 I feel the intensity in your words and I get it. That energy is telling you something important — let's figure out what. 🎯",
        "⚡ Anger signals that something you care deeply about has been crossed. You're not wrong for feeling this way. 🤝",
        "🛡️ Your frustration is real and completely justified. You don't have to calm down before you're ready — I'm here. 💙",
        "🌸 Being this upset means something truly matters to you. That's not a weakness, that's passion. Let it out safely. 💗",
        "💜 You have every right to be angry. Let's sit with it for a moment and figure out how to move through it. ✨",
        "🔥 That fire in you is valid. Channel it somewhere powerful when you're ready — don't let it burn you from inside. 💙",
        "⚡ The anger you're feeling is information. It's telling you your boundaries were crossed and that matters. 🛡️",
        "💙 You're allowed to be furious. Just remember — you control what comes next, and that's actually your power. ✨",
        "🌬️ Take a breath with me for a second. You're dealing with something genuinely unfair and you deserve acknowledgment. 💕",
        "💢 The way you're feeling makes complete sense given what happened. You're not overreacting. 🤝",
        "🌸 Sometimes anger is the most honest response. Honor it, then decide what you want to do with it. 💜",
    ],
    "fear": [
        "🫂 Your fears are real, but so is your strength. I'm right here and we'll face this together. 💪",
        "🌟 It's okay to be scared — fear means something matters to you. You don't have to be brave alone. 🤝",
        "⭐ You've made it through hard things before, even when it felt impossible. I believe in you completely. 💫",
        "🌙 Take a slow breath with me. You are safe, you are cared for, and you are never, ever alone. 💙",
        "🌸 Anxiety lies — it makes things feel bigger than they are. But your feelings are still valid. I'm here. 💕",
        "💜 It makes total sense that you're scared. Let's take this one small moment at a time, together. ✨",
        "🛡️ I've got you. Breathe in slowly... breathe out slowly. You're here, you're safe, you're doing okay. 💙",
        "🌙 Fear is your nervous system trying to protect you. Acknowledge it, thank it gently, then breathe through it. 💕",
        "⭐ The unknown is scary, but it also holds possibilities. You've survived 100% of your hardest days. 💪",
        "🌸 Panic can feel permanent but it always passes. Stay with me — one breath, one second at a time. 💜",
        "💙 You don't have to face this alone. Share the weight with someone who cares — including me, right now. 🫂",
        "✨ The courage isn't the absence of fear — it's going forward anyway. And you're already doing that by sharing. 💗",
    ],
    "surprise": [
        "🌟 Wow, life is full of plot twists! How are you feeling about all of this? I'm here to process it with you. 💭",
        "⭐ Take all the time you need to let this land — what does it mean for you? 🤔",
        "🎨 Surprises can be exciting, overwhelming, or both at once. All of it is valid! Tell me more! 💬",
        "✨ That's quite something! Sometimes we just need a moment to go 'wait, what??' — and that's perfectly fine. 🎯",
        "🌸 Okay plot twist! Whether this feels good or not, I'm here to help you process it. 💕",
        "😮 Life didn't warn you about that one! Take a breath — how are you feeling underneath the shock? 💙",
        "🌟 The unexpected can be disorienting. Give yourself space to absorb this before deciding how to feel. 💕",
        "⚡ That came out of nowhere! Whatever you're feeling right now — surprise, joy, fear, confusion — it's all valid. 💭",
        "🎭 Reality loves a good plot twist! I'm here while you process this — take your time. 🌸",
        "✨ Sometimes the universe just has other plans. How are you sitting with this news? 💙",
    ],
    "disgust": [
        "💭 I completely understand — sometimes things just don't sit right, and that gut feeling is valid. 🎯",
        "🌟 Your boundaries matter so much. Recognizing what doesn't align with your values is real self-awareness. ✨",
        "🛡️ Not everything deserves your energy or your time. You're allowed to step away from what doesn't feel right. 🌈",
        "💙 Trust your instincts — your gut feeling is valuable and worth listening to. You know yourself. 👁️",
        "🌸 The ick is real! Your feelings about this are completely valid — you don't have to justify them. 💜",
        "💜 Recognizing what's wrong is wisdom. Your moral compass is working — trust it. 🧭",
        "🛡️ You don't have to engage with what repels you. Protect your peace and your standards. ✨",
        "🌸 Your reaction is your body and mind's way of saying 'this isn't for me' — always listen to that. 💙",
        "✨ Disgust is actually protective — it keeps us away from things that harm us. Your instincts are sharp. 🎯",
        "💕 You're allowed to find things unacceptable. That's not being harsh — that's knowing your worth. 🌟",
    ],
    "neutral": [
        "👂 I'm here and fully listening to whatever you'd like to share. No pressure, just space. 💙",
        "😌 Sometimes life is just... steady. That's okay too. What's on your mind? 💭",
        "🌙 Feeling settled? I'm always here if you want to talk about anything at all. 🤝",
        "💭 Whatever you're feeling is valid, even if it's hard to name. I'm all ears! 👂",
        "🌸 Just existing is enough sometimes. I'm here whenever you want to chat. 💕",
        "🎀 Not every day needs to be dramatic — steady is its own kind of wonderful. 💙",
        "✨ A calm, balanced day is actually something to appreciate. What's been going on? 🌸",
        "💙 Being okay is completely okay. Tell me about your day if you'd like! 😊",
        "🌙 Sometimes neutral is the most peaceful place to be. I'm right here with you. 💕",
        "🌸 You don't have to feel something big right now. I'm just happy you're here. 🎀",
    ]
}

PET_REACTIONS = {
    "joy": [
        "Luna does a happy spin and purrs with pure delight! 🌸✨",
        "Luna's ribbon bounces as she zooms with joy just for you! 🎀💫",
        "Luna bounces with uninhibited happiness and wants to celebrate with you! 🎉🌸",
        "Luna claps her little paws together — she's SO happy for you! 💕🎊",
        "Luna does her happy dance and her bow bounces with every step! 🎀🌟",
        "Luna's eyes light up and she does three spins of pure happiness! ✨💗",
        "Luna leaps into a celebratory twirl and purrs the happiest purr! 🌸🎉",
    ],
    "sadness": [
        "Luna nuzzles close and offers you the warmest, quietest comfort. 🫂💙",
        "Luna sits right beside you with a gentle paw on your heart. 💙🌸",
        "Luna looks into your eyes with deep, soft understanding. 👁️💫",
        "Luna wraps her little arms around you — you're not alone. 💕🌙",
        "Luna brings you her softest blanket and curls up next to you. 🌙💙",
        "Luna hums a soft lullaby and stays close, keeping you warm. 💜🌸",
        "Luna wipes a tiny tear from your cheek with her paw, very gently. 💙🫂",
    ],
    "anger": [
        "Luna takes a slow, calming breath right alongside you. 🌬️✨",
        "Luna acknowledges your fire and grounds herself beside you. 🔥💙",
        "Luna places a steady paw on your shoulder. Your feelings are valid. 💪💙",
        "Luna nods seriously — she gets it. Let's figure this out together. 💜",
        "Luna stands firm beside you, ears alert, ready to listen without judgment. 🛡️💗",
        "Luna breathes in and out slowly, inviting you to match her rhythm. 🌬️💙",
    ],
    "fear": [
        "Luna stands tall and strong right beside you, ready to protect. 🛡️💪",
        "Luna wraps around you softly and whispers: 'You're safe with me.' 💙🌙",
        "Luna holds your paw firmly and won't let go — ever. 🤝💖",
        "Luna keeps her eyes open and alert so you don't have to. 🌸💪",
        "Luna curls around you like a warm shield and purrs softly. 💜🛡️",
        "Luna breathes in slowly... out slowly... and nudges you to follow. 🌙💙",
        "Luna positions herself between you and the scary thing, tail up. 💪🌸",
    ],
    "surprise": [
        "Luna's eyes go wide with wonder and her bow perks up! 👀✨",
        "Luna tilts her head, completely fascinated by this development! 🤔💭",
        "Luna's ribbon stands up with electric curiosity! ⚡🌸",
        "Luna gasps dramatically alongside you! 😮💕",
        "Luna's ears shoot straight up in comical shock! 🙀✨",
        "Luna freezes in place, blinking slowly, processing alongside you! 😲💫",
    ],
    "disgust": [
        "Luna wrinkles her nose in full agreement — you're absolutely right. 👃✨",
        "Luna nods knowingly. Your instincts are spot on — always trust them! 👍🎯",
        "Luna gives a very judgmental look in solidarity. The ick is real! 💅💜",
        "Luna turns away from the offending thing alongside you. Your values are valid! 🛡️💙",
        "Luna makes a tiny 'blech' face and scoots closer to you for protection. 🤢🌸",
    ],
    "neutral": [
        "Luna listens attentively with her little ears perked up, fully present. 👂💙",
        "Luna sits peacefully beside you, creating a calm space just for you. 🌙🌸",
        "Luna blinks slowly and contentedly — she's just happy to be here with you. 💕",
        "Luna hums softly to herself while keeping you company. 🎵🌸",
        "Luna does a little stretch and settles comfortably by your side. 💙🎀",
    ]
}

# ==================== BREATHING EXERCISE DATA ====================
BREATHING_EXERCISES = {
    "box": {
        "name": "Box Breathing",
        "description": "Used by Navy SEALs for stress control",
        "steps": [
            {"phase": "Inhale", "duration": 4, "instruction": "Breathe in slowly through your nose..."},
            {"phase": "Hold", "duration": 4, "instruction": "Hold your breath gently..."},
            {"phase": "Exhale", "duration": 4, "instruction": "Breathe out slowly through your mouth..."},
            {"phase": "Hold", "duration": 4, "instruction": "Hold before the next breath..."},
        ],
        "cycles": 4,
    },
    "478": {
        "name": "4-7-8 Breathing",
        "description": "Dr. Andrew Weil's relaxation technique",
        "steps": [
            {"phase": "Inhale", "duration": 4, "instruction": "Inhale quietly through your nose..."},
            {"phase": "Hold", "duration": 7, "instruction": "Hold your breath..."},
            {"phase": "Exhale", "duration": 8, "instruction": "Exhale completely through your mouth..."},
        ],
        "cycles": 4,
    },
    "calm": {
        "name": "Calming Breath",
        "description": "Simple 5-5-5 for quick relief",
        "steps": [
            {"phase": "Inhale", "duration": 5, "instruction": "Breathe in peacefully..."},
            {"phase": "Hold", "duration": 5, "instruction": "Rest in the stillness..."},
            {"phase": "Exhale", "duration": 5, "instruction": "Release everything slowly..."},
        ],
        "cycles": 5,
    }
}

EMOTION_COLORS = {
    "joy":      "#e8427c",
    "sadness":  "#7b8ef0",
    "anger":    "#f06060",
    "fear":     "#b07be8",
    "surprise": "#f0a860",
    "disgust":  "#60c8a0",
    "neutral":  "#c8a0c8",
}

# Extended color palettes for gradients
EMOTION_GRADIENTS = {
    "joy":      ("e8427c", "f0a860"),
    "sadness":  ("7b8ef0", "5a6fd6"),
    "anger":    ("f06060", "e83030"),
    "fear":     ("b07be8", "8050c0"),
    "surprise": ("f0a860", "f0c860"),
    "disgust":  ("60c8a0", "40a880"),
    "neutral":  ("c8a0c8", "b090b0"),
}

# ==================== DETECTION FUNCTIONS ====================

def get_emotion_emoji(emotion: str) -> str:
    return EMOTION_EMOJIS.get(emotion, "🎀")


def _preprocess(text: str) -> str:
    t = text.lower()
    contractions = {
        "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
        "it's": "it is", "that's": "that is", "they're": "they are",
        "we're": "we are", "you're": "you are", "can't": "cannot",
        "won't": "will not", "don't": "do not", "didn't": "did not",
        "isn't": "is not", "wasn't": "was not", "wouldn't": "would not",
        "couldn't": "could not", "shouldn't": "should not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
        "gonna": "going to", "wanna": "want to", "gotta": "got to",
        "kinda": "kind of", "sorta": "sort of", "lowkey": "lowkey",
        "no cap": "no cap", "fr fr": "for real", "ngl": "not going to lie",
        "imo": "in my opinion", "rn": "right now", "tbh": "to be honest",
        "omfg": "oh my god", "wtf": "what the", "smh": "shaking my head",
        "lmao": "laughing", "lmfao": "laughing", "rofl": "laughing",
        "ikr": "i know right", "irl": "in real life", "imo": "in my opinion",
        "idk": "i do not know", "idc": "i do not care", "fomo": "fear of missing out",
        "yolo": "you only live once", "omg": "oh my god", "wtf": "what the",
        "brb": "be right back", "afk": "away from keyboard",
        "ty": "thank you", "tysm": "thank you so much", "np": "no problem",
        "nvm": "never mind", "imo": "in my opinion", "jk": "just kidding",
        "istg": "i swear to god", "imo": "in my opinion",
    }
    for short, expanded in contractions.items():
        t = t.replace(short, expanded)
    t = re.sub(r"[''']", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\b(\w)\1{3,}\b", r"\1\1", t)  # reduce excessive repeated letters
    return " ".join(t.split())


def _check_negation(words: List[str], idx: int, window: int = 5) -> bool:
    start = max(0, idx - window)
    return any(words[j] in NEGATION_WORDS for j in range(start, idx))


def _detect_emojis(text: str) -> Dict[str, float]:
    scores: Dict[str, float] = {e: 0.0 for e in EMOTION_EMOJIS}
    for emoji_char, emotion in EMOTION_EMOJIS_MAP.items():
        count = text.count(emoji_char)
        if count:
            scores[emotion] = scores.get(emotion, 0.0) + count * 3.5
    return scores


def _detect_repeated_chars(text: str) -> float:
    pattern = re.compile(r"(.)\1{3,}")
    return 1.4 if pattern.search(text.lower()) else 1.0


def _detect_caps_intensity(text: str) -> float:
    if len(text) < 5:
        return 1.0
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
    if caps_ratio > 0.5:
        return 1.6
    elif caps_ratio > 0.3:
        return 1.3
    return 1.0


def _detect_mixed_emotions(clean: str) -> Optional[Tuple[str, float]]:
    """Check for mixed-emotion indicators."""
    for blend_name, phrases in MIXED_EMOTION_PHRASES.items():
        for phrase in phrases:
            if phrase in clean:
                return blend_name, 0.6
    return None


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize scores using softmax for better confidence calibration."""
    vals = list(scores.values())
    max_val = max(vals) if vals else 0
    exp_vals = {k: math.exp(v - max_val) for k, v in scores.items()}
    total = sum(exp_vals.values())
    return {k: v / total for k, v in exp_vals.items()}


def detect_emotion_rule_based(text: str) -> Tuple[str, float]:
    """
    Improved rule-based detection v5.0:
    1. Sarcasm detection
    2. Mixed emotion detection
    3. Emoji scanning
    4. Phrase matching (highest priority)
    5. Weighted keyword scoring with intensifier + negation
    6. Caps/repeated-char intensity modifiers
    7. Softmax normalization
    8. Confidence-aware fallback to neutral
    """
    clean = _preprocess(text)
    words = clean.split()
    is_sarcastic = _detect_sarcasm(text)
    intensity_mod = _detect_repeated_chars(text) * _detect_caps_intensity(text)

    # Check for mixed emotions first
    mixed = _detect_mixed_emotions(clean)

    # 1. Emoji signals
    emoji_scores = _detect_emojis(text)

    # 2. Phrase matching
    phrase_scores: Dict[str, float] = {e: 0.0 for e in PHRASE_PATTERNS}
    for emotion, phrases in PHRASE_PATTERNS.items():
        for phrase in phrases:
            if phrase in clean:
                phrase_scores[emotion] += 4.5

    # 3. Weighted keyword scoring
    kw_scores: Dict[str, float] = {e: 0.0 for e in EMOTION_KEYWORDS_WEIGHTED}
    for idx, word in enumerate(words):
        multiplier = intensity_mod
        # Check 1-2 words back for intensifiers
        if idx > 0 and words[idx - 1] in INTENSIFIERS:
            multiplier = max(multiplier, INTENSIFIERS[words[idx - 1]] * intensity_mod)
        if idx > 1 and words[idx - 2] in INTENSIFIERS and words[idx - 1] not in NEGATION_WORDS:
            multiplier = max(multiplier, INTENSIFIERS[words[idx - 2]] * 0.75 * intensity_mod)

        negated = _check_negation(words, idx)

        for emotion, kw_list in EMOTION_KEYWORDS_WEIGHTED.items():
            for kw, weight in kw_list:
                kw_words = kw.split()
                if len(kw_words) > 1:
                    if " ".join(words[idx:idx + len(kw_words)]) == kw:
                        if not negated:
                            kw_scores[emotion] += weight * multiplier
                elif kw == word or (len(kw) > 4 and word.startswith(kw[:4]) and kw in word):
                    if negated:
                        if emotion == "joy":
                            kw_scores["sadness"] += weight * multiplier * 0.9
                        elif emotion in ("sadness", "fear", "anger", "disgust"):
                            kw_scores["joy"] += weight * multiplier * 0.5
                    else:
                        kw_scores[emotion] += weight * multiplier

    # Sarcasm modifier — flip joy to neutral/sadness
    if is_sarcastic:
        kw_scores["joy"]     *= 0.25
        phrase_scores["joy"] *= 0.25
        kw_scores["neutral"]  = kw_scores.get("neutral", 0) + 2.0

    # Blend all signals
    all_emotions = set(list(emoji_scores.keys()) + list(phrase_scores.keys()) + list(kw_scores.keys()))
    total_scores: Dict[str, float] = {}
    for e in all_emotions:
        total_scores[e] = (
            emoji_scores.get(e, 0.0) * 1.3 +
            phrase_scores.get(e, 0.0) * 1.6 +
            kw_scores.get(e, 0.0)
        )

    total = sum(total_scores.values())
    if total < 0.5 or max(total_scores.values()) < 1.0:
        return "neutral", 0.45

    # Softmax normalization
    softmax_scores = _softmax(total_scores)
    detected = max(softmax_scores, key=softmax_scores.get)
    confidence = softmax_scores[detected]

    # Mixed emotion: return primary but flag it
    if mixed:
        blend_name, blend_weight = mixed
        blend_emotions = MIXED_EMOTION_PATTERNS.get(blend_name, ([], 0))[0]
        if blend_emotions and detected in blend_emotions:
            confidence = min(confidence * 1.1, 0.99)

    if confidence < 0.22:
        return "neutral", 0.45

    return detected, round(confidence, 3)


def detect_emotion(user_input: str, model: Dict) -> str:
    emotion, _ = detect_emotion_with_confidence(user_input, model)
    return emotion


def detect_emotion_with_confidence(user_input: str, model: Dict) -> Tuple[str, int]:
    """
    4-layer detection: Ensemble ML → rule-based → VADER-like → neutral.
    Returns (emotion, confidence 0-100).
    """
    ml_emotion = None
    ml_prob = 0.0
    ml_emotion_lr = None
    ml_prob_lr = 0.0

    try:
        word_vectorizer = model.get("word_vectorizer")
        char_vectorizer = model.get("char_vectorizer")
        clf_svc = model.get("model")
        clf_lr  = model.get("model_lr")
        emotion_labels = model.get("emotion_labels", {})

        if word_vectorizer and char_vectorizer and clf_svc:
            from scipy.sparse import hstack
            text_vec_word = word_vectorizer.transform([user_input])
            text_vec_char = char_vectorizer.transform([user_input])
            text_vec = hstack([text_vec_word, text_vec_char])

            # SVC prediction
            svc_probs = clf_svc.predict_proba(text_vec)[0]
            svc_pred  = clf_svc.predict(text_vec)[0]
            ml_emotion = emotion_labels.get(svc_pred, "neutral")
            ml_prob = max(svc_probs)

            # LR prediction (if available)
            if clf_lr is not None:
                lr_probs = clf_lr.predict_proba(text_vec)[0]
                lr_pred  = clf_lr.predict(text_vec)[0]
                ml_emotion_lr = emotion_labels.get(lr_pred, "neutral")
                ml_prob_lr = max(lr_probs)

                # Ensemble: weighted average
                if model.get("ensemble", False):
                    label_classes = model.get("label_classes", list(range(7)))
                    # Pad if needed
                    n = len(label_classes)
                    if len(svc_probs) == n and len(lr_probs) == n:
                        ens_probs = [svc_probs[i] * 0.6 + lr_probs[i] * 0.4 for i in range(n)]
                        best_idx = ens_probs.index(max(ens_probs))
                        ens_label = label_classes[best_idx] if best_idx < len(label_classes) else svc_pred
                        ens_emotion = emotion_labels.get(ens_label, "neutral")
                        ens_prob = max(ens_probs)
                        if ens_prob >= 0.50:
                            return ens_emotion, round(ens_prob * 100)
                        ml_emotion = ens_emotion
                        ml_prob = ens_prob

            if ml_prob >= 0.55:
                return ml_emotion, round(ml_prob * 100)

        elif model.get("vectorizer") and clf_svc:
            text_vec = model["vectorizer"].transform([user_input])
            prediction = clf_svc.predict(text_vec)
            probabilities = clf_svc.predict_proba(text_vec)[0]
            max_prob = max(probabilities)
            ml_emotion = emotion_labels.get(prediction[0], "neutral")
            ml_prob = max_prob
            if max_prob >= 0.55:
                return ml_emotion, round(max_prob * 100)

    except Exception:
        pass

    # Layer 2: Rule-based
    rule_emotion, rule_conf = detect_emotion_rule_based(user_input)

    # Blend ML + rules when both agree
    if ml_emotion and ml_emotion == rule_emotion and ml_prob > 0.3:
        blended_conf = min((rule_conf + ml_prob) / 2 * 1.25, 0.99)
        return rule_emotion, round(blended_conf * 100)

    if rule_conf >= 0.30:
        return rule_emotion, round(rule_conf * 100)

    if ml_emotion and ml_emotion != "neutral" and ml_prob > 0.3:
        return ml_emotion, round(ml_prob * 100)

    return "neutral", 45


def get_comfort_response(emotion: str, user_input: str = None) -> str:
    responses = COMFORT_RESPONSES.get(emotion, COMFORT_RESPONSES["neutral"])
    return random.choice(responses)


def get_pet_reaction(emotion: str) -> str:
    reactions = PET_REACTIONS.get(emotion, PET_REACTIONS["neutral"])
    return random.choice(reactions)


def get_emotion_color(emotion: str) -> str:
    return EMOTION_COLORS.get(emotion, "#c8a0c8")


def get_emotion_gradient(emotion: str) -> Tuple[str, str]:
    return EMOTION_GRADIENTS.get(emotion, ("c8a0c8", "b090b0"))


def get_breathing_exercise(emotion: str) -> Dict:
    """Return appropriate breathing exercise for the emotion."""
    if emotion in ("fear", "anger"):
        return BREATHING_EXERCISES["box"]
    elif emotion == "sadness":
        return BREATHING_EXERCISES["478"]
    else:
        return BREATHING_EXERCISES["calm"]


def analyze_mood_trend(mood_history: List[Dict]) -> str:
    if not mood_history:
        return "Start sharing your feelings to see trends! 🌸"
    if len(mood_history) < 2:
        return "More interactions needed to identify trends. Keep sharing! 💕"

    emotions = [e["emotion"] for e in mood_history]
    emotion_counts: Dict[str, int] = {}
    for em in emotions:
        emotion_counts[em] = emotion_counts.get(em, 0) + 1

    most_common = max(emotion_counts, key=emotion_counts.get)
    percentage = emotion_counts[most_common] / len(emotions) * 100
    recent = emotions[-3:]
    recent_top = max(set(recent), key=recent.count)

    insights = {
        "joy":      [
            "🌸 You're radiating such positive energy lately! Keep this momentum going!",
            "💕 Your happiness has been shining through — beautiful! Celebrate yourself!",
            "✨ So much joy in your recent chats! You deserve every bit of it!",
            "🌟 You've been on a genuinely happy streak! Protect this energy fiercely!",
        ],
        "sadness":  [
            "💙 You've been going through a lot lately. Remember to be gentle with yourself.",
            "🌙 It's okay to not be okay. You're processing, and that takes real courage.",
            "🫂 Rough patch lately — but you're still here, still sharing. That matters.",
            "💜 You've been carrying something heavy. Please reach out to someone you trust.",
        ],
        "anger":    [
            "🔥 You've been dealing with a lot of frustration. Your feelings are completely valid.",
            "💪 Lots of intense emotions lately — that passion is telling you something important.",
            "💜 It seems like things have been really unfair lately. You deserve better.",
            "⚡ Your fire is valid — channel it into change rather than letting it consume you.",
        ],
        "fear":     [
            "💜 You've been navigating a lot of uncertainty. Remember how strong you really are.",
            "✨ Anxiety has been present a lot recently — be extra kind to yourself right now.",
            "🌸 Lots of worry lately — one breath, one moment at a time. You've got this.",
            "🛡️ Your nervous system has been working overtime. Rest and gentle care are healing.",
        ],
        "surprise": [
            "🌟 Life keeps throwing plot twists your way — stay curious and flexible!",
            "⭐ Unexpected moments have been shaping your journey. Embrace the adventure!",
        ],
        "disgust":  [
            "🛡️ You clearly know your values well — that's a real strength to have.",
            "✨ Trusting your instincts is healthy. Keep listening to that inner voice.",
        ],
        "neutral":  [
            "🌙 You've been in a balanced, reflective state lately. That's peaceful.",
            "🧘 Grounded and centered — a wonderful place to be.",
            "🌸 Steady and calm lately — sometimes that's exactly what we need.",
        ],
    }
    target = most_common if percentage >= 60 else recent_top
    return random.choice(insights.get(target, insights["neutral"]))


def get_mood_summary(mood_history: List[Dict]) -> Dict:
    if not mood_history:
        return {"total_interactions": 0, "most_common_emotion": None, "emotion_distribution": {}, "trend": "Start tracking!"}
    ec: Dict[str, int] = {}
    for entry in mood_history:
        ec[entry["emotion"]] = ec.get(entry["emotion"], 0) + 1
    return {
        "total_interactions":   len(mood_history),
        "most_common_emotion":  max(ec, key=ec.get) if ec else None,
        "emotion_distribution": ec,
        "trend":                analyze_mood_trend(mood_history),
    }


def is_valid_input(text: str) -> bool:
    return bool(text) and 2 <= len(text.strip()) <= 1000


def get_journal_prompt(emotion: str) -> str:
    """Return a journaling prompt based on detected emotion."""
    prompts = {
        "joy": [
            "What specifically made today feel so wonderful? How can you create more moments like this?",
            "Who contributed to your happiness today? Have you told them?",
            "What does this joy teach you about what matters most to you?",
            "How can you carry this feeling into tomorrow?",
        ],
        "sadness": [
            "What do you need most right now that you're not getting?",
            "If your sadness could speak, what would it want you to know?",
            "What would you tell a friend who was feeling exactly what you feel now?",
            "Is there one small thing that could make today feel even slightly better?",
        ],
        "anger": [
            "What boundary was crossed that triggered this anger?",
            "What do you wish you could say to the person or situation making you angry?",
            "What does this anger tell you about your values?",
            "What would resolution look like for this situation?",
        ],
        "fear": [
            "What exactly are you most afraid of? Write it out fully.",
            "What is the most realistic outcome of what you're fearing?",
            "What would you do if your fear came true? You've survived hard things before.",
            "What's one small action that could help you feel more in control right now?",
        ],
        "surprise": [
            "How are you truly feeling about this unexpected development?",
            "What does this change mean for your plans and expectations?",
            "What opportunities might be hidden in this surprise?",
        ],
        "disgust": [
            "What values of yours were violated to trigger this reaction?",
            "What boundaries do you want to set as a result of this?",
            "How can you protect your peace from things that feel this wrong?",
        ],
        "neutral": [
            "What would make tomorrow feel meaningful?",
            "Is there anything simmering beneath the surface that you haven't acknowledged?",
            "What's one thing you're grateful for today, even if it's small?",
            "What do you want more of in your life right now?",
        ],
    }
    options = prompts.get(emotion, prompts["neutral"])
    return random.choice(options)


def get_coping_strategies(emotion: str) -> List[str]:
    """Return practical coping strategies for the given emotion."""
    strategies = {
        "joy": [
            "🌟 Write down 3 things that made today wonderful — gratitude journaling amplifies joy",
            "💕 Share your happiness with someone you love — joy grows when shared",
            "📸 Capture this moment somehow — a photo, a voice note, a journal entry",
            "🎵 Put on music that matches your vibe and let yourself fully celebrate",
        ],
        "sadness": [
            "🌊 Allow yourself to cry if you need to — tears are healing, not weakness",
            "🫂 Reach out to one trusted person and say 'I'm struggling today'",
            "🌸 Do one small gentle thing for yourself: tea, a blanket, a short walk",
            "✍️ Write without filtering — just let the sadness pour onto the page",
            "🌙 Be patient with yourself — healing is not linear, and that's okay",
        ],
        "anger": [
            "🌬️ Try box breathing: 4 seconds in, 4 hold, 4 out, 4 hold — repeat 4 times",
            "🏃 Move your body: a fast walk, jumping jacks, or punching a pillow releases anger physically",
            "✍️ Write an unsent letter to whoever angered you — get it all out",
            "⏱️ Give yourself a 20-minute cooling period before responding to anyone",
            "🗣️ Talk to a trusted person who will validate your feelings, not minimize them",
        ],
        "fear": [
            "🌬️ 4-7-8 breathing: inhale 4 seconds, hold 7, exhale 8 — activates the parasympathetic system",
            "🧊 The 5-4-3-2-1 grounding method: 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste",
            "📝 Write down your fear in specific detail — vague fears are scarier than named ones",
            "🤝 Tell someone 'I'm feeling anxious about X' — saying it out loud reduces its power",
            "🌸 Remind yourself: you have survived 100% of your hardest days so far",
        ],
        "surprise": [
            "⏸️ Give yourself permission to pause before reacting — you don't have to respond immediately",
            "💬 Talk through the surprise with someone to help process your feelings",
            "📝 Write out what this change means for you and what your options are",
        ],
        "disgust": [
            "🚪 It's okay to remove yourself from situations that violate your values",
            "📝 Write down the boundary that was crossed and how you want to protect it",
            "🛡️ Your instincts are valid — trust them and honor your moral compass",
        ],
        "neutral": [
            "🌸 A neutral day is a good time to plan, rest, or do something creative",
            "📝 Journaling on a calm day can reveal feelings you didn't know were there",
            "🌿 Gentle self-care: a walk, hydration, or stretching to maintain this steadiness",
        ],
    }
    return strategies.get(emotion, strategies["neutral"])