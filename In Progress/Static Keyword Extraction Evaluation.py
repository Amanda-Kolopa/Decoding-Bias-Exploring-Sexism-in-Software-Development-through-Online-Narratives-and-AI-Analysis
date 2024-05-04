################################################################################
# Format the Data to Evaluate Static Keyword Extraction
# Developers Experiences of Sexism
# By: Amanda Kolopanis
# ! Code is still a work in progress - ** Please use as a guideline ** !
################################################################################
import pandas as pd

keywords_per_category = {
    'feminine-coded goods and services': {
        'cool', 'natural', 'healthy', 'loyal', 'good', 'affection', 'adoration', 'indulgence', 'loving', 'acceptance',
        'nurturing', 'safety', 'security', 'safe haven', 'kindness', 'compassion', 'mortal attention', 'concern',
        'soothing', 'caring', 'trust', 'respect', 'attentive', 'relationship', 'giving', 'awesome', 'wonderful',
        'lovely', 'great', 'excellent', 'beautiful', 'terrific', 'fantastic', 'fabulous', 'superb', 'hot', 'marvelous',
        'stellar', 'fine', 'neat', 'prime', 'heavenly', 'calm', 'genuine', 'unaffected', 'simple', 'honest', 'innocent',
        'naïve', 'sincere', 'pure', 'raw', 'organic', 'wholesome', 'easy', 'strong', 'fit', 'hearty', 'active',
        'lively', 'faithful', 'dependable', 'devoted', 'trustworthy', 'trusty', 'dedicated', 'reliable', 'pleasant',
        'positive', 'favorable', 'valuable', 'noble', 'decent', 'ethical', 'mortal', 'auspicious', 'happy', 'sentiment',
        'liking', 'veneration', 'blessing', 'privilege', 'courtesy', 'leniency', 'permissiveness', 'admiring',
        'affectionate', 'amiable', 'adoring', 'passionate', 'approval', 'support', 'embracing', 'adoption', 'female',
        'feminine', 'matronly', 'womanly', 'parental', 'protection', 'safeguards', 'safeness', 'guard', 'safekeeping',
        'shield', 'goodwill', 'grace', 'kindliness', 'benevolence', 'gentleness', 'sweetness', 'kindheartedness',
        'benignity', 'empathy', 'sympathy', 'mercy', 'pity', 'commiseration', 'worry', 'fear', 'anxiety', 'unease',
        'concernment', 'relaxing', 'comforting', 'tranquilizing', 'calming', 'hypnotic', 'quieting', 'sedative',
        'dreamy', 'peaceful', 'restful', 'reassuring', 'compassionate', 'benevolent', 'helpful', 'sympathetic',
        'thoughtful', 'generous', 'humane', 'kindly', 'warm', 'soft', 'sensitive', 'tender', 'responsive', 'receptive',
        'considerate', 'warmhearted', 'tenderhearted', 'softhearted', 'nice', 'confide', 'confidence', 'faith',
        'assurance', 'entrustment', 'credence', 'depend on', 'count on', 'admiration', 'regard', 'appreciation',
        'praise', 'recognition', 'reverence', 'kind', 'respectful', 'solicitous', 'gracious', 'polite', 'connection',
        'association', 'kinship', 'relation', 'linkage', 'affiliation', 'interaction', 'bond', 'communication',
        'friendship', 'bestowing', 'offering'
    },

    'testimonial injustice': {
        'catcalling', 'trolling', 'condescending', 'mansplain', 'moralizing', 'blaming', 'silencing', 'lampooning',
        'satirizing', 'sexualizing', 'desexualizing', 'belittling', 'caricaturing', 'exploiting', 'erasing',
        'infantilizing', 'ridiculing', 'humiliating', 'mocking', 'slurring', 'vilifying', 'demonizing', 'shunning',
        'shaming', 'patronizing', 'dismissive', 'disparaging', 'less credible', 'less competent', 'accused', 'impugned',
        'convicted', 'corrected', 'diminished', 'outperformed', 'jeering', 'hooting', 'snorting', 'sniffing', 'jibing',
        'gibing', 'sneering', 'laughing', 'whistle', 'heckling', 'holler', 'bossy', 'impudent', 'snooty', 'lecturing',
        'preaching', 'condemning', 'condemn', 'condemned', 'faulting', 'denouncing', 'knocking', 'attacking',
        'slamming', 'censuring', 'suppressing', 'quelling', 'subduing', 'censor', 'muffling', 'spoofing', 'burlesquing',
        'mimicking', 'banter', 'bitterness', 'cynicism', 'minimizing', 'discounting', 'derogating', 'pejorative',
        'contemptuous', 'contempt', 'deride', 'scoff', 'taunt', 'tease', 'parodying', 'imitating', 'abuse',
        'manipulate', 'misuse', 'eradicating', 'destroying', 'abolishing', 'obliterating', 'immaturity', 'ignorance',
        'childishness', 'derisive', 'baiting', 'deriding', 'fooling', 'mortifying', 'demeaning', 'embarrassing',
        'degrading', 'ignominious', 'humbling', 'uncivil', 'sarcastic', 'satirical', 'disrespectful', 'sardonic',
        'negativistic', 'disgrace', 'insinuate', 'affronting', 'blaspheming', 'cursing', 'berating', 'insulting',
        'offensive', 'rude', 'abusive', 'malign', 'smearing', 'libeling', 'slandering', 'defaming', 'discrediting',
        'diabolize', 'torment', 'affliction', 'avoidance', 'ostracism', 'exile', 'isolation', 'rejection', 'expulsion',
        'evasion', 'disgracing', 'dishonoring', 'abasement', 'mortification', 'deceiving', 'groveling', 'grudging',
        'domineering', 'dominant', 'disdainful', 'authoritarian', 'snobbish', 'dismissing', 'denigrating',
        'bad-mouthing', 'derogative', 'defamatory', 'deprecatory', 'incompetent', 'unskillful', 'helpless',
        'inadequate', 'incapable', 'unqualified', 'useless', 'inept', 'unfit', 'inexperienced', 'indicted', 'charged',
        'blamed', 'prosecuted', 'censured', 'criticized', 'denounced', 'appealed', 'castigated', 'reprobate', 'guilty',
        'culpable', 'punishable', 'rectified', 'amended', 'revised', 'culprit', 'imprison', 'rebuke', 'discipline',
        'reprimand', 'chide', 'admonish', 'assessed', 'scorn', 'devalue', 'denigrate', 'decry', 'deprecate',
        'depreciate', 'derogate', 'beat', 'exceed', 'surpass', 'outdo', 'defeated', 'bested'
    },

    'gendered split perception': {
        'duplicitous', 'vindictive', 'conniving', 'untrustworthy', 'careless', 'shady', 'crooked', 'rule-breaker',
        'dangerous', 'suspicious', 'risky', 'deceptive', 'deceitful', 'dishonest', 'fraudulent', 'malicious',
        'vengeful', 'vicious', 'revengeful', 'petty', 'spiteful', 'merciless', 'resentful', 'scheming', 'plotting',
        'conspiring', 'collusive', 'shifty', 'disloyal', 'unreliable', 'untrusty', 'devious', 'unfaithful',
        'thoughtless', 'reckless', 'sloppy', 'negligent', 'indifferent', 'unconcerned', 'absent-minded', 'unthinking',
        'cursory', 'inconsiderate', 'unmindful', 'incautious', 'impetuous', 'unwary', 'mindless', 'dubious',
        'questionable', 'unscrupulous', 'dodgy', 'suspect', 'fishy', 'disreputable', 'suborned', 'corrupt',
        'dishonorable', 'troubling', 'perilous', 'precarious', 'ugly', 'unsafe', 'unstable', 'alarming', 'menacing',
        'insecure', 'irresponsible', 'distrustful', 'skeptical', 'mistrustful', 'unusual', 'unbelieving', 'leery',
        'hazardous', 'threatening', 'dicey', 'misleading', 'sneaky', 'spurious', 'ambiguous', 'delusive', 'fallacious',
        'delusory', 'beguiling'
    },

    'social dominance penalty': {
        'smother', 'intimidate', 'powerful women', 'powerful woman', 'threatening', 'underestimate', 'doubt',
        'victim blaming', 'crazy', 'hysterical', 'disliked', 'rejected', 'hostile', 'abrasive', 'manipulative',
        'arrogant', 'aggressive', 'ballbreaker', 'castrating bitch', 'punished', 'real woman', 'real women', 'bitch',
        'witch', 'unfair', 'rigid', 'cold', 'psychotic', 'overwhelm', 'stifle', 'repress', 'hold back', 'restrain',
        'bottle up', 'bully', 'frighten', 'scare', 'coerce', 'startle', 'browbeat', 'harass', 'bulldoze', 'pressure',
        'terrify', 'hound', 'daunt', 'oppress', 'constrain', 'dishearten', 'dismay', 'ominous', 'intimidatory',
        'terrorizing', 'sinister', 'underrate', 'undervalue', 'minimize', 'disbelief', 'hesitation', 'uncertainty',
        'skepticism', 'kooky', 'mad', 'nuts', 'nutty', 'silly', 'wacky', 'ridiculous', 'absurd', 'foolish', 'ludicrous',
        'mental', 'irrational', 'agitated', 'distraught', 'frantic', 'frenzied', 'neurotic', 'convulsive', 'upset',
        'hatred', 'disgust', 'hostility', 'loath', 'disapproval', 'distaste', 'animosity', 'aversion', 'antagonism',
        'displeasure', 'antipathy', 'enmity', 'animus', 'disinclination', 'repugnance', 'detestation', 'abhor',
        'detest', 'execrated', 'despised', 'abandoned', 'deserted', 'disused', 'denied', 'disregarded', 'dumped',
        'ditched', 'rebuff', 'antagonistic', 'mean', 'hateful', 'inhospitable', 'nasty', 'unfavorable', 'unfriendly',
        'catty', 'sour', 'inimical', 'negative', 'irritating', 'annoying', 'harsh', 'cruel', 'unpleasant', 'rough',
        'unkind', 'frustrating', 'disturbing', 'aggravating', 'bothersome', 'deceive', 'shrewd', 'vain', 'smug',
        'pompous', 'imperious', 'cocky', 'conceited', 'cavalier', 'bumptious', 'assumptive', 'pretentious',
        'belligerent', 'combative', 'destructive', 'intrusive', 'assertive', 'malevolent', 'pushy', 'pugnacious',
        'penalized', 'fined', 'sentenced', 'chastised', 'levied', 'floozy', 'harlot', 'hussy', 'slut', 'tart', 'tramp',
        'vamp', 'wench', 'whore', 'broad', 'hellion', 'termagant', 'vixen', 'hag', 'shrew', 'foul', 'shameful',
        'biased', 'prejudiced', 'discriminatory', 'strict', 'rigorous', 'stern', 'stringent', 'aloof', 'distant',
        'frigid', 'apathetic', 'glacial', 'demented', 'insane', 'unhinged', 'lunatic', 'paranoid', 'psycho', 'maniac'
    },
}


def count_matching_keywords(text, keywords):
    return sum(text.lower().count(keyword.lower()) for keyword in keywords)

############################# Combine csv files #############################
print("Phase 1: Combining CSV Data Files")
dfs = []
for key in keywords_per_category:
    df = pd.read_csv('C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/cleaned text data files/'+key+'.csv')
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

########################## Count matching keywords ##########################
print("Phase 2: Counting Keywords per Text")
df["Keyword Count"] = df.apply(
    lambda row: count_matching_keywords(
        row["Text"], keywords_per_category.get(row["Potential Category"], [])), axis=1
)

df.to_csv('C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/In Progress/Counted SKE Evaluation.csv', index=False)