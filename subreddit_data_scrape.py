################################################################################
# Scraping 2000 Hot and Top from SubReddits using Keyword Extraction
# By: Amanda Kolopanis
# ! Code is still a work in progress - ** Please use as a guideline ** !
# ! Please excuse the explicit language !
################################################################################

import praw  # required library for Reddit
import numpy as np
import re
from datetime import datetime
import csv
import os

# Log-in credentials to access Reddit
# Use this link to get appropriate information to run following script: https://www.honchosearch.com/blog/seo/how-to-use-praw-and-crawl-reddit-for-subreddit-post-data/
# TODO: Fill in the blanks as required
reddit = praw.Reddit(
    client_id="5_BSfXajkZuIosfiDHJtZw",
    client_secret="b-C6YATBvugw_u75y3YSPRw6B-aznQ",
    password="9P-H$TbDc97tP!e",
    user_agent="AK_data_scrape/0.0.1",
    username="Desperate_Cell_2377"
)

# Built a dictionary containing relevant keywords per category (this is specific for my thesis)
# TODO: update categories and associated keywords/phrases to fit your research
keywords_per_category = {
    'context-related': {'sexist', 'sexism', 'misogyny', 'misogynistic', 'discrimination', 'discriminate',
                        'gender bias', 'prejudice'},

    'benevolent sexism': {'care', 'carefulness', 'solicitude', 'solicitousness', 'consideration', 'lovingness', 'loving', 'thoughtfulness', 'considerateness', 'pampering', 'coddling', 'babying', 'responsibility',
                          'concern', 'worry', 'fear', 'anxiety', 'unease', 'concernment', 'respect', 'admiration', 'regard', 'appreciation', 'praise', 'recognition', 'reverence', 'love', 'adore', 'cherish', 'warmth',
                          'tenderness', 'endearment', 'comfort', 'acceptance', 'approval', 'support', 'embracing', 'adoption', 'kindness', 'goodwill', 'grace', 'kindliness', 'benevolence', 'gentleness', 'sweetness',
                          'kindheartedness', 'benignity', 'affection', 'sentiment', 'liking', 'adoration', 'respect', 'veneration', 'compassion', 'empathy', 'sympathy', 'mercy', 'pity', 'commiseration',
                          'mortal attention', 'safety', 'safe', 'protection', 'safeguards', 'safeness', 'guard', 'security', 'safekeeping', 'shield', 'safe haven', 'real woman', 'stereotypes', 'pigeonhole',
                          'typecast', 'generalization', 'baddie', 'vamp', 'babe', 'baby', 'bae', 'chick', 'doll', 'jezebel', 'sweetie', 'sweetheart', 'sweet',
                          'honey', 'hon', 'jade', 'buttercup', 'fox', 'snow bunny', 'bombshell', 'maternal', 'real ladies', 'real lady', 'real women', 'real girl', 'gentlewoman', 'gentlewomen', 'girly', 'barbie', 'nurture',
                          'giving', 'bestowing', 'offering', 'caring', 'compassionate', 'benevolent', 'helpful', 'sympathetic', 'concerned', 'thoughtful', 'generous', 'humane', 'kindly', 'warm', 'soft', 'sensitive', 'tender',
                          'responsive', 'receptive', 'considerate', 'warmhearted', 'tenderhearted', 'softhearted', 'nice', 'loving', 'admiring', 'affectionate', 'amiable', 'adoring', 'passionate', 'attentive',
                          'kind', 'respectful', 'solicitous', 'gracious', 'polite', 'cool', 'awesome', 'wonderful', 'lovely', 'great', 'excellent', 'beautiful', 'terrific', 'fantastic', 'fabulous', 'superb', 'hot', 'marvelous',
                          'stellar', 'fine', 'neat', 'prime', 'heavenly', 'calm', 'natural', 'genuine', 'unaffected', 'simple', 'honest', 'innocent', 'naïve', 'sincere', 'pure', 'raw', 'organic', 'wholesome', 'easy',
                          'healthy', 'strong', 'fit', 'hearty', 'active', 'lively', 'well', 'loyal', 'faithful', 'dependable', 'devoted', 'trustworthy', 'trusty', 'trust', 'dedicated', 'reliable', 'good', 'pleasant', 'positive',
                          'favorable', 'valuable', 'noble', 'decent', 'ethical', 'mortal', 'auspicious', 'happy', 'indulgence', 'blessing', 'privilege', 'courtesy', 'leniency', 'permissiveness', 'nurturing',
                          'female', 'feminine', 'matronly', 'womanly', 'parental', 'soothing', 'relaxing', 'comforting', 'tranquilizing', 'calming', 'hypnotic', 'quieting', 'sedative', 'dreamy', 'peaceful', 'restful',
                          'reassuring', 'motherly', 'womanly', 'hormonal', 'emotional', 'soft nature', 'soft-nature', 'polite', 'accommodating', 'clean', 'sensitive', 'soft-spoken', 'dainty', 'fragile', 'gentle',
                          'graceful', 'pleasant', 'delicate', 'giddy'},

    'misogynist hostility': {'ballbreaker', 'castrating bitches', 'bad women', 'bitch', 'witch', 'slut', 'whore', 'cow', 'pussy', 'pussies', 'thot', 'hoe', 'cunt', 'tramp', 'vixen', 'butterface', 'twat', 'puss', 'ho', 'hoochie',
                             'shrew', 'she-devil', 'wench', 'minx', 'coquette','tease', 'hag', 'termagant', 'two-bagger', 'quean', 'harlot', 'mistress', 'concubine', 'trollop', 'tail', 'piece of tail', 'piece of ass', 'skank',
                             'cocktease', 'cock-tease', 'cock tease', 'prick-tease', 'prick tease', 'temptress', 'seductress', 'maneater', 'man-eater', 'old bat', 'mantrap', 'strumpet', 'virago', 'harridan', 'doxy', 'harridan',
                             'doxy', 'femme fatale', 'milf', 'gilf', 'cougar', 'call girl', 'lady of the evening', 'woman of the street', 'bimbo', 'floozy', 'hussy', 'nag', 'loose', 'ditzy', 'ditsy', 'frigid', 'frumpy', 'shrill',
                             'hysterical',  'bimbo',  'scatterbrain', 'birdbrain', 'airhead', 'feather brain', 'blonde', 'smother', 'overwhelm', 'stifle', 'suppress', 'repress', 'penalizing', 'fining', 'disciplining', 'criticizing',
                             'hold back', 'restrain', 'bottle up', 'intimidate', 'bully', 'frighten', 'scare', 'coerce', 'startle', 'browbeat', 'harass', 'bulldoze', 'pressure', 'terrify', 'hound', 'daunt',
                             'moralizing', 'lecturing', 'preaching', 'blaming', 'condemning', 'condemn', 'condemned', 'faulting', 'denouncing', 'knocking', 'attacking', 'slamming', 'censuring', 'punishing',
                             'sentencing', 'chastising', 'convicting', 'silencing', 'suppressing', 'quelling', 'subduing', 'repressing', 'censor', 'muffling', 'lampooning', 'spoofing', 'burlesquing',
                             'mimicking', 'banter', 'bitterness', 'cynicism', 'satirizing', 'sexualizing', 'desexualizing', 'belittling', 'minimizing', 'discounting', 'derogating', 'pejorative', 'contemptuous', 'contempt',
                             'caricaturing', 'deride', 'scoff', 'taunt', 'tease', 'parodying', 'imitating', 'exploiting', 'corrupt', 'abuse', 'manipulate', 'misuse', 'erasing', 'eradicating', 'destroying', 'abolishing',
                             'obliterating', 'evincing', 'displaying', 'revealing', 'betraying', 'infantilizing', 'immaturity', 'ignorance', 'childishness', 'ridiculing', 'derisive', 'baiting', 'deriding', 'fooling',
                             'humiliating', 'mortifying', 'demeaning', 'embarrassing', 'degrading', 'ignominious', 'humbling', 'mocking', 'uncivil', 'sarcastic', 'satirical', 'disrespectful', 'sardonic', 'negativistic',
                             'slurring', 'disgrace', 'insinuate', 'affronting', 'blaspheming', 'cursing', 'berating', 'vilifying', 'insulting', 'offensive', 'rude', 'abusive', 'malign', 'smearing', 'libeling',
                             'slandering', 'defaming', 'discrediting', 'demonizing', 'diabolize', 'torment', 'affliction', 'shunning', 'avoidance', 'ostracism', 'exile', 'isolation', 'rejection', 'expulsion', 'evasion',
                             'shaming', 'disgracing', 'dishonoring', 'abasement', 'mortification', 'deceiving', 'groveling', 'grudging', 'patronizing', 'domineering', 'dominant', 'disdainful', 'authoritarian', 'snobbish',
                             'dismissive', 'disparaging', 'dismissing', 'denigrating', 'bad-mouthing', 'derogative', 'defamatory', 'deprecatory', 'arrogant', 'vain', 'smug', 'pompous', 'imperious', 'cocky', 'conceited',
                             'cavalier', 'bumptious', 'assumptive', 'pretentious', 'aggressive', 'hostile', 'belligerent', 'combative', 'destructive', 'intrusive', 'assertive', 'malevolent', 'pushy', 'pugnacious',
                             'condescending', 'bossy', 'impudent', 'snooty', 'indifference', 'unconcern', 'insensitivity', 'negligence', 'mean', 'inferior', 'callous', 'unfair', 'foul', 'nasty', 'shameful', 'biased',
                             'prejudiced', 'discriminatory', 'rigid', 'strict', 'rigorous', 'stern', 'stringent', 'cold', 'aloof', 'distant', 'frigid', 'apathetic', 'glacial', 'psychotic', 'demented', 'insane',
                             'unhinged', 'lunatic', 'paranoid', 'psycho', 'maniac', 'frail', 'hypersensitive', 'infantile', 'childish'},

    'misogynistic gatekeeping': {'moralism', 'puritanism', 'prudery', 'morality', 'prudishness', 'effigies', 'dummy', 'puppet', 'likeness', 'statue', 'scapegoats', 'victims', 'excuses', 'pushover', 'stooge', 'sucker', 'sacrifice', 'patsy', 'weakling',
                                 'doormat', 'leadership', 'management', 'governance', 'leaders', 'supervisors', 'chiefs', 'directors', 'guidance', 'lead', 'authority', 'dominion', 'control', 'sway', 'command', 'dominance', 'police', 'prerogative', 'force',
                                 'jurisdiction', 'rule', 'influence', 'leverage', 'clout', 'important', 'money', 'cash', 'coin', 'funds', 'dollar', 'wealth', 'wage', 'salary', 'power', 'energy', 'strength', 'capability', 'ability', 'social status', 'prestige',
                                 'fame', 'dignity', 'esteem', 'importance', 'prominence', 'stature', 'status', 'renown', 'notoriety', 'significance', 'rank', 'level', 'position', 'echelon', 'pride', 'pridefulness', 'confidence', 'ego', 'self-respect', 'honor',
                                 'congratulate', 'reputation', 'repute', 'credit', 'character', 'rule-breaker', 'damsel in distress', 'housewife', 'housewives', 'homemaking', 'homemaker', 'womanly duty', 'womanly duties', 'girly girl', 'girly-girl',
                                 'hoyden', 'ladette', 'lady of the house', 'woman thing', 'catcalling', 'jeering', 'hooting', 'snorting', 'sniffing', 'jibing', 'gibing', 'sneering', 'laughing', 'whistle', 'heckling', 'holler', 'trolling', 'mansplaining',
                                 'lashing out', 'wishful thinking', 'willful denial', 'accused', 'indicted', 'charged', 'blamed', 'prosecuted', 'censured', 'impugned', 'criticized', 'denounced', 'appealed', 'castigated', 'reprobated', 'convicted', 'guilty',
                                 'culpable', 'punishable', 'corrected', 'rectified', 'amended', 'revised', 'culprit', 'imprison', 'rebuke', 'discipline', 'reprimand', 'chide', 'admonish', 'assessed', 'diminished', 'belittle', 'scorn', 'devalue', 'denigrate', 'decry',
                                 'deprecate', 'depreciate', 'derogate', 'outperformed', 'beat', 'exceed', 'surpass', 'outdo', 'defeated', 'bested', 'underestimate', 'underrate', 'undervalue', 'minimize', 'victim blaming', 'punished', 'penalized', 'fined', 'sentenced',
                                 'chastised', 'levied', 'rejected', 'abandoned', 'deserted', 'disused', 'denied', 'disregarded', 'dumped', 'ditched', 'rebuff', 'abrasive', 'irritating', 'annoying', 'harsh', 'bitter', 'cruel', 'unpleasant', 'rough', 'unkind', 'frustrating',
                                 'disturbing', 'aggravating', 'bothersome', 'threatening', 'menacing', 'ominous', 'intimidatory', 'terrorizing', 'sinister', 'powerful woman', 'powerful women', 'less competent', 'incompetent', 'unskillful', 'helpless',
                                 'inadequate', 'incapable', 'unqualified', 'useless', 'inept', 'unfit', 'inexperienced', 'duplicitous', 'deceptive', 'deceitful', 'dishonest', 'fraudulent', 'shady', 'crooked', 'crazy', 'kooky', 'mad', 'nuts', 'nutty', 'silly', 'wacky',
                                 'ridiculous', 'absurd', 'foolish', 'ludicrous', 'mental', 'irrational', 'hysterical', 'agitated', 'distraught', 'frantic', 'frenzied', 'neurotic', 'convulsive', 'upset', 'vindictive', 'malicious', 'vengeful', 'vicious', 'revengeful',
                                 'petty', 'spiteful', 'merciless', 'resentful', 'manipulativeness', 'manipulative', 'exploit', 'deceive', 'devious', 'shrewd', 'coldness', 'detachment', 'objectivity', 'cold', 'frigidness', 'aggression', 'hostility', 'defiance',
                                 'belligerence', 'malice', 'antagonism', 'antipathy', 'malevolence', 'pugnacity', 'encroachment', 'ladylike', 'womanlike'},

    # 'intellectual inferiority': ['stupid', 'bimbo', 'dumb', 'dummy', 'fool', 'idiot', 'moron', 'slow', 'silly',
    #                              'brainless', 'ditsy', 'ditzy', 'smart for a girl', 'smart for a woman', 'scatterbrain',
    #                              'birdbrain', 'airhead', 'featherbrain', 'blonde', 'blond', 'dumbass', 'dopey',
    #                              'unintelligent', 'mindless', 'empty-headed', 'brain-dead', 'foolish', 'blockhead',
    #                              'bonehead', 'imbecile', 'imprudent', 'vacuous', 'dippy', 'dizzy', 'dingbat',
    #                              'airbrain', 'rattlebrain', 'space case', 'fluffhead', 'ninny', 'ditz', 'numskull',
    #                              'knucklehead', 'dunce', 'thickhead', 'dope', 'dimwit', 'donkey', 'retard',
    #                              'incompetent', 'barbie', 'bimbette', 'dumbo', 'imbecilic', 'halfwitted', 'halfwit',
    #                              'half witted', 'simple minded', 'simple-minded', 'damfool', 'cretinous', 'cretin',
    #                              'dolt', 'lunkhead', 'dullard', 'dunderhead', 'pillock', 'witless', 'batty', 'fatuous',
    #                              'dappy', 'senseless', 'braindead', 'lame-brained', 'weak minded', 'weak-minded',
    #                              'illogical', 'tomfool', 'thoughtless', 'nonsensical', 'buffoon', 'chump', 'jackass',
    #                              'nincompoop', 'simpleton', 'oaf', 'hardhead', 'clodpoll', 'asinine', 'inane',
    #                              'lightheaded', 'nitwit', 'pinhead', 'clod', 'boob', 'dummkopf', 'knothead', 'dodo',
    #                              'ignoramus', 'meathead', 'clueless', 'muttonhead', 'loggerhead', 'booby', 'knobhead',
    #                              'puerile', 'infantile', 'childish', 'gormless', 'puerile', 'numpty', 'doofus',
    #                              'dingus'],
}

intellectual_inferiority = {'stupid', 'bimbo', 'dumb', 'dummy', 'fool', 'idiot', 'moron', 'slow', 'silly',
                           'brainless', 'ditsy', 'ditzy', 'smart for a girl', 'smart for a woman', 'scatterbrain',
                           'birdbrain', 'airhead', 'featherbrain', 'blonde', 'blond', 'dumbass', 'dopey',
                           'unintelligent', 'mindless', 'empty-headed', 'brain-dead', 'foolish', 'blockhead',
                           'bonehead', 'imbecile', 'imprudent', 'vacuous', 'dippy', 'dizzy', 'dingbat',
                           'airbrain', 'rattlebrain', 'space case', 'fluffhead', 'ninny', 'ditz', 'numskull',
                           'knucklehead', 'dunce', 'thickhead', 'dope', 'dimwit', 'donkey', 'retard',
                           'incompetent', 'barbie', 'bimbette', 'dumbo', 'imbecilic', 'halfwitted', 'halfwit',
                           'half witted', 'simple minded', 'simple-minded', 'damfool', 'cretinous', 'cretin',
                           'dolt', 'lunkhead', 'dullard', 'dunderhead', 'pillock', 'witless', 'batty', 'fatuous',
                           'dappy', 'senseless', 'braindead', 'lame-brained', 'weak minded', 'weak-minded',
                           'illogical', 'tomfool', 'thoughtless', 'nonsensical', 'buffoon', 'chump', 'jackass',
                           'nincompoop', 'simpleton', 'oaf', 'hardhead', 'clodpoll', 'asinine', 'inane',
                           'lightheaded', 'nitwit', 'pinhead', 'clod', 'boob', 'dummkopf', 'knothead', 'dodo',
                           'ignoramus', 'meathead', 'clueless', 'muttonhead', 'loggerhead', 'booby', 'knobhead',
                           'puerile', 'infantile', 'childish', 'gormless', 'puerile', 'numpty', 'doofus',
                           'dingus'}
# TODO: input the name of the subReddits you want to scrape here
subreddits = ['girlsgonewired', 'womenintech', 'womenwhocode', 'chickswhocode', 'ladydevs',
              'ladycoders', 'cswomen', 'xxstem', 'lesbiancoders', 'pyladies', 'launchcodergirl']

# Initialize variable that holds all scraped content
scraped_data = []


def review_intellectual_inferiority_category_post(current_subreddit, hot_or_top, current_post):

    for keyword in intellectual_inferiority:

        # Search for each word in the current post and it's title
        if re.search(r"\b{}\b".format(keyword), current_post.selftext,
                     re.IGNORECASE) is not None or \
                re.search(r"\b{}\b".format(keyword), current_post.title,
                          re.IGNORECASE) is not None:

            # Save author's name (or [deleted])
            if current_post.author is not None:
                author_name = current_post.author.name
            else:
                author_name = '[deleted]'

            # Add the post to the scraped dataset
            # This saves the:
            #    - subReddit name
            #    - category (hot or top)
            #    - the scraped date and time
            #    - the posted date and time
            #    - author's name
            #    - 'Post' to describe the data (post or comment)
            #    - post title
            #    - the post ID
            #    - 'NULL' to represent no image (still in progress)
            #    - Affiliated category (based on my thesis)
            scraped_data.append([current_subreddit, hot_or_top, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                     datetime.fromtimestamp(current_post.created_utc), author_name,
                                     'Post', current_post.title, current_post.id, 'NULL',
                                     current_post.selftext,
                                     'intellectual inferiority'])
            # else:
            #     scraped_data.append([current_subreddit, hot_or_new, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            #                          datetime.fromtimestamp(current_post.created_utc), author_name,
            #                          'Comment', current_post.title, current_post.id,
            #                          'NULL', current_post.body, 'intellectual inferiority'])


def review_intellectual_inferiority_category_comment(current_subreddit, hot_or_top, current_comment, post_title):

    for keyword in intellectual_inferiority:

        # Search for each word in the current post and it's title
        if re.search(r"\b{}\b".format(keyword), current_comment.body, re.IGNORECASE) \
                is not None:

            # Save author's name (or [deleted])
            if current_comment.author is not None:
                author_name = current_comment.author.name
            else:
                author_name = '[deleted]'

            # Add the comment to the scraped dataset
            scraped_data.append([current_subreddit, hot_or_top, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                 datetime.fromtimestamp(current_comment.created_utc), author_name,
                                 'Comment', post_title, current_comment.id,
                                 'NULL', current_comment.body, 'intellectual inferiority'])


def scrape_subreddit(current_subreddit, post_type, limit):

    print('Current SubReddit: {}'.format(current_subreddit))

    # Counter to see progress of scraping
    sanity_check = 1

    # Loop through each HOT post (current limit = 2000 - this could be modified if needed)
    if post_type == 'hot':
        reddit_post_type = reddit.subreddit(current_subreddit).hot(limit=limit)
    else:
        reddit_post_type = reddit.subreddit(current_subreddit).top(limit=limit)

    # Loop through each post
    for post_submission in reddit_post_type:
        try:
            print('At {} post number: {} of 2000'.format(post_type, sanity_check))

            # Loop through each category containing the keywords
            for category_keywords_tuple in keywords_per_category.items():

                # Loop through the current category's keywords
                for keyword_per_category in category_keywords_tuple[1]:

                    # Search for each word in the current post and it's title
                    if re.search(r"\b{}\b".format(keyword_per_category), post_submission.selftext,
                                 re.IGNORECASE) is not None or \
                            re.search(r"\b{}\b".format(keyword_per_category), post_submission.title,
                                      re.IGNORECASE) is not None:

                        # Save the author's name of the post The purpose of having an author's name as '[deleted]' is
                        # for previous Reddit users that closed their accounts
                        if post_submission.author is not None:
                            author_name = post_submission.author.name
                        else:
                            author_name = '[deleted]'

                        # Add the post to the scraped dataset
                        # This saves the:
                        #    - subReddit name
                        #    - category (hot or top)
                        #    - the scraped date and time
                        #    - the posted date and time
                        #    - author's name
                        #    - 'Post' to describe the data (post or comment)
                        #    - post title
                        #    - the post ID
                        #    - 'NULL' to represent no image (still in progress)
                        #    - Affiliated category (based on my thesis)
                        scraped_data.append([current_subreddit, post_type, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                             datetime.fromtimestamp(post_submission.created_utc), author_name,
                                             'Post', post_submission.title, post_submission.id, 'NULL',
                                             post_submission.selftext,
                                             category_keywords_tuple[0]])

                        #review_intellectual_inferiority_category_post(current_subreddit, post_type, post_submission)

            # Checks if the post is a URL before scraping comments
            # Reason: the URL needs to be associated to a post to access its comments
            # Note: this part throws some errors but the script will still run (work in progress)
            if 'png' not in post_submission.url:

                # Retrieve all the comments of the current post
                submission = reddit.submission(url='https://www.reddit.com{}'.format(post_submission.url))
                submission.comments.replace_more(limit=None, threshold=0)
                all_comments = submission.comments.list()

                # Loop through each comment of the current post
                for comment in all_comments:
                    try:

                        # Loop through each category containing the keywords
                        for category_keywords_tuple in keywords_per_category.items():

                            # Loop through the current category's keywords
                            for keyword_per_category in category_keywords_tuple[1]:

                                # Search for each word in the current comment
                                if re.search(r"\b{}\b".format(keyword_per_category), comment.body, re.IGNORECASE) \
                                        is not None:

                                    # Save author's name (or [deleted])
                                    if comment.author is not None:
                                        author_name = comment.author.name
                                    else:
                                        author_name = '[deleted]'

                                    # Add the comment to the scraped dataset
                                    scraped_data.append([current_subreddit, post_type, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                         datetime.fromtimestamp(comment.created_utc), author_name,
                                                         'Comment', post_submission.title, comment.id,
                                                         'NULL', comment.body, category_keywords_tuple[0]])

                                    #review_intellectual_inferiority_category_comment(current_subreddit, post_type,
                                    #                                                 comment, post_submission.title)

                    # Throws exception when the URL is an image or a link to another website
                    except Exception as e:
                        print('Comment exception: {}'.format(e))
                        continue

            # Increment the counter for scraping progress
            sanity_check += 1

        # Throws exception when the URL is an image or a link to another website
        except Exception as e:
            print('Post exception: {}'.format(e))
            sanity_check += 1
            continue

    # This is the same process for NEW posts (improvements in progress to remove duplicate code)
    # sanity_check = 1
    # for post_submission in reddit.subreddit(subreddit).new(limit=2000):
    #     try:
    #         print('At NEW post number: {} of 2000'.format(sanity_check))
    #
    #         # Loop through the categories
    #         for category_keywords_tuple in keywords_per_category.items():
    #
    #             # Loop through the keywords per category
    #             for keyword_per_category in category_keywords_tuple[1]:
    #
    #                 # Search for each word in the post
    #                 if re.search(r"\b{}\b".format(keyword_per_category), post_submission.selftext,
    #                              re.IGNORECASE) is not None or \
    #                         re.search(r"\b{}\b".format(keyword_per_category), post_submission.title,
    #                                   re.IGNORECASE) is not None:
    #                     author_name = post_submission.author.name if post_submission.author.name is not None else 'None'
    #
    #                     # Add the post to the scraped dataset
    #                     scraped_data.append([subreddit, 'Top', datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #                                          datetime.fromtimestamp(post_submission.created_utc), author_name,
    #                                          'Post', post_submission.title, post_submission.id, 'NULL',
    #                                          post_submission.selftext,
    #                                          category_keywords_tuple[0]])
    #
    #
    #         if 'png' not in post_submission.url:
    #             submission = reddit.submission(url='https://www.reddit.com{}'.format(post_submission.url))
    #             submission.comments.replace_more(limit=None, threshold=0)
    #             all_comments = submission.comments.list()
    #
    #             for comment in all_comments:
    #                 try:
    #
    #                     # Loop through the categories
    #                     for category_keywords_tuple in keywords_per_category.items():
    #
    #                         # Loop through the keywords per category
    #                         for keyword_per_category in category_keywords_tuple[1]:
    #
    #                             # Search for each word in the comment
    #                             if re.search(r"\b{}\b".format(keyword_per_category), comment.body, re.IGNORECASE) \
    #                                     is not None:
    #                                 author_name = comment.author.name if not None else 'None'
    #
    #                                 # Add the comment to the scraped dataset
    #                                 scraped_data.append([subreddit, 'Top', datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #                                                      datetime.fromtimestamp(comment.created_utc), author_name,
    #                                                      'Comment', post_submission.title, comment.id,
    #                                                      'NULL', comment.body, category_keywords_tuple[0]])
    #                 except Exception as e:
    #                     print('Comment exception: {}'.format(e))
    #                     continue
    #
    #         # update post progress checker
    #         sanity_check += 1
    #
    #     except Exception as e:
    #         print('Post exception: {}'.format(e))
    #         sanity_check += 1
    #         continue


for subreddit in subreddits:
    scrape_subreddit(subreddit, 'hot', 2000)
    scrape_subreddit(subreddit, 'top', 2000)

# Supposed to remove any duplicates before adding to file (need to double check if this works or not tbh)
for data in scraped_data:
    if any(data[7] and data[10] in inner_list for inner_list in scraped_data):
        scraped_data.remove(data)

# Displays the number of posts and comments scraped from subReddits
print(len(scraped_data))


# Headers for the csv file (as previously outlined)
csv_headers = ['SubReddit', 'Hot or Top', 'Scraped Date', 'Posted Date', 'Posted by', 'Post or Comment', 'Post Title',
               'ID', 'Image Ref', 'Text',
               'Potential Category']

# Saves today's date to name the folder to be created
today_folder_name = datetime.now().strftime("%Y-%m-%d")


# Attempts to create a folder named after today's date
# try:
#     if os.path.isdir('scraped data/' + today_folder_name) is False:
#         os.mkdir('scraped data/' + today_folder_name)
# except Exception as e:
#     print('Directory error: {}'.format(e))

# Saves the scraped data to a csv file
# TODO: add the path to where you would like the scraped data to be stored
with open(
        'C:/Users/amand/OneDrive/Desktop/Thesis/hot_top_2000.csv',
        'w', newline='', encoding="utf-8") as f:
    write = csv.writer(f)
    write.writerow(csv_headers)
    write.writerows(scraped_data)


print('File should be in folder!')
