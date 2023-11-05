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

# TODO: input the name of the subReddits you want to scrape here
subreddits = ['girlsgonewired', 'womenintech', 'womenwhocode', 'chickswhocode', 'ladydevs',
              'ladycoders', 'cswomen', 'xxstem', 'lesbiancoders', 'pyladies', 'launchcodergirl']

# Initialize variable that holds all scraped content
scraped_data = []


def scrape_subreddit(current_subreddit, post_type, limit):

    print('Current SubReddit: {}'.format(current_subreddit))

    # Counter to see progress of scraping
    sanity_check = 1

    # Loop through each hot/top/new post (current limit = 2000 - this could be modified if needed)
    if post_type == 'hot':
        reddit_post_type = reddit.subreddit(current_subreddit).hot(limit=limit)
    elif post_type == 'top':
        reddit_post_type = reddit.subreddit(current_subreddit).top(limit=limit)
    else:
        reddit_post_type = reddit.subreddit(current_subreddit).new(limit=limit)

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


for subreddit in subreddits:
    scrape_subreddit(subreddit, 'hot', 2000)
    scrape_subreddit(subreddit, 'top', 2000)
    scrape_subreddit(subreddit, 'new', 2000)

# Supposed to remove any duplicates before adding to file
for data in scraped_data:
    if any(data[7] and data[10] in inner_list for inner_list in scraped_data):
        scraped_data.remove(data)

# Displays the total number of posts and comments scraped from all subReddits
print(len(scraped_data))


# Headers for the csv file (as previously outlined)
csv_headers = ['SubReddit', 'Hot or Top', 'Scraped Date', 'Posted Date', 'Posted by', 'Post or Comment', 'Post Title',
               'ID', 'Image Ref', 'Text',
               'Potential Category']


# Saves the scraped data to a csv file
# TODO: add the path to where you would like the scraped data to be stored
with open(
        'C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/OG_hot_top_new_2000.csv',
        'w', newline='', encoding="utf-8") as f:
    write = csv.writer(f)
    write.writerow(csv_headers)
    write.writerows(scraped_data)


print('File should be in folder!')
