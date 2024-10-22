from urllib.parse import urlparse, parse_qs
import pandas as pd
import math
from collections import Counter
import re
import requests
import threading
import pandas as pd

# Example of English character distribution (frequencies based on a corpus)
english_frequencies = {
    'e': 0.1202, 't': 0.0910, 'a': 0.0812, 'o': 0.0768, 'i': 0.0731,
    'n': 0.0695, 's': 0.0628, 'r': 0.0602, 'h': 0.0592, 'd': 0.0432,
    'l': 0.0398, 'u': 0.0288, 'c': 0.0271, 'm': 0.0261, 'f': 0.0230,
    'y': 0.0211, 'w': 0.0209, 'g': 0.0203, 'p': 0.0182, 'b': 0.0149,
    'v': 0.0111, 'k': 0.0069, 'x': 0.0017, 'q': 0.0011, 'j': 0.0010,
    'z': 0.0007
}

suspicious_words = [
    'sweepstakes', 'prize', 'win', 'free', 'cash', 'bonus', 'reward', 'claim', 'voucher', 'gift',
    'credit', 'money', 'discount', 'offer', 'guaranteed', 'urgent', 'verify', 'secure', 'account',
    'login', 'password', 'phishing', 'scam', 'fraud', 'spoof', 'alert', 'warning', 'hack', 'virus',
    'malware', 'infection', 'compromised', 'breach', 'suspicious', 'unusual', 'confidential',
    'admin', 'cmd', 'confirm', 'delivery', 'dhl', 'fedex', 'financial', 'invoice', 'irs', 'logon',
    'n', 'no', 'notification', 'postal', 'post', 'signin', 'statement', 'ticket', 'update', 'ups',
    'usps'
]

punctuation_symbols = ['.', '!', '&', ',', '#', '$', '%']

popular_brands = [
    "google", "facebook", "youtube", "twitter", "amazon",
    "linkedin", "ebay", "instagram", "yahoo", "wikipedia",
    "apple", "netflix", "microsoft", "samsung", "sony",
    "alibaba", "tencent", "hilton", "marriott", "bmw",
    "mercedesbenz", "toyota", "nike", "adidas",
    "intel", "amd", "huawei", "xiaomi", "baidu",
    "walmart", "costco", "target", "homedepot", "lowes", "uniqlo", "zara",
    "visa", "mastercard", "paypal", "jpmorganchase", "bankofamerica",
    "hbomax", "hulu", "espn", "bbc",
    "underarmour", "puma", "lululemon",
    "cocacola", "pepsico", "nestle", "starbucks", "mcdonalds",
    "tesla", "volkswagen", "hyundai", "honda", "ford",
    "chanel", "louisvuitton", "gucci", "prada", "dior",
    "rolex", "cartier", "tiffany&co",
    "ritzcarlton", "intercontinental", "hyatt",
    "disney", "spotify", "twitch", "telegram",
    "jd", "otto", "rakuten",
    "playstation", "xbox", "nintendo",
    "realmadrid", "barcelona", "manchesterunited", "newyorkyankees", "goldenstatewarriors"
]

def frequncies_for_KL(url):

    def calculate_character_probabilities(url):
        # Count character frequencies
        character_counts = Counter(url)
        total_characters = sum(character_counts.values())

        # Calculate probabilities
        character_probabilities = {char: count / total_characters for char, count in character_counts.items()}

        return character_probabilities

    def kl_divergence(p, q):
        # Compute KL divergence
        return sum(p[char] * math.log(p[char] / q.get(char, 1e-9)) for char in p)
    
    character_probabilities = calculate_character_probabilities(url)
    # Normalize English frequencies to probabilities
    total_english_characters = sum(english_frequencies.values())
    english_probabilities = {char: freq / total_english_characters for char, freq in english_frequencies.items()}

    # Compute KL divergence
    kl_div = kl_divergence(character_probabilities, english_probabilities)
    return kl_div

def calculate_entropy(s):
    # Count character frequencies
    character_counts = Counter(s)
    total_characters = len(s)
    
    # Calculate probabilities
    probabilities = {char: count / total_characters for char, count in character_counts.items()}
    
    # Calculate entropy
    entropy = -sum(p * math.log2(p) for p in probabilities.values() if p > 0)
    
    return entropy

def calculate_digit_letter_ratio(url):
    # Count letters and digits in the URL
    num_letters = sum(c.isalpha() for c in url)
    num_digits = sum(c.isdigit() for c in url)
    
    # Calculate the ratio of digits to letters
    if num_letters > 0:
        digit_letter_ratio = num_digits / num_letters
    else:
        digit_letter_ratio = 0.0  # Handle division by zero (no letters)
    
    return digit_letter_ratio

def count_top_level_domains_in_path(url):
    # Extract the path component from the URL
    parsed_url = urlparse(url)
    
    # Get the path component from the parsed URL
    path = parsed_url.path
    
    # Split the path into segments and filter out empty segments
    path_segments = [seg for seg in path.split('/') if seg]
    
    # Initialize a counter for top-level domains (TLDs) in the path
    tld_count = 0
    
    # Iterate through each path segment
    for segment in path_segments:
        # Check if the segment resembles a TLD (e.g., ends with a known TLD)
        if '.' in segment:  # If segment contains a dot
            last_part = segment.split('.')[-1]  # Get the last part after the dot
            # Check if the last part is a valid TLD (using a simple check for demonstration)
            if len(last_part) > 1:  # Valid TLDs typically have more than one character
                tld_count += 1  # Increment the TLD count
    
    # Determine if the URL is likely suspicious based on the count of TLDs in the path
    return tld_count

def count_dashes_in_path(url):
    # Extract the path component from the URL
    parsed_url = urlparse(url)
    path = parsed_url.path
    
    # Count the occurrences of dashes ('-') in the path
    dash_count = path.count('-')
    
    return dash_count
def urllength(url):
    return  len(url)

def has_digits_in_domain(url):
    """
    Check if the domain part of the URL contains any digits.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc  # Extract domain from URL

    # Check if the domain contains any digits
    if any(char.isdigit() for char in domain):
        return 1
    else:
        return 0

def count_suspicious_words(url, suspicious_words):
    """
    Count the occurrences of suspicious words in the URL.
    """
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()  # Extract path from URL and convert to lowercase
    query = parsed_url.query.lower()  # Extract query parameters from URL and convert to lowercase

    # Combine path and query for comprehensive analysis
    url_text = path + '?' + query

    total_count = 0

    # Count occurrences of suspicious words
    for word in suspicious_words:
        word_count = url_text.count(word)
        total_count += word_count

    return total_count

def count_subdomains(url):
    """
    Count the number of sub-domains in the URL's domain part.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()  # Extract domain from URL and convert to lowercase

    # Count the number of dots (.) in the domain
    subdomain_count = domain.count('.')
    
    return subdomain_count







def is_brand_name_modified(url, popular_brands):
    """
    Check if the URL's domain name contains a modified version of a popular brand name.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()  # Extract domain from URL and convert to lowercase

    # Iterate over each popular brand name
    for brand in popular_brands:
        brand_lower = brand.lower()

        # Check for modified brand names with prefixes or suffixes
        if '-' in domain:
            parts = domain.split('-')
            if parts[0] == brand_lower or parts[-1] == brand_lower:
                return 1

    return 0

def is_long_hostname_phishy(url, threshold=22):
    """
    Check if the length of the URL's hostname (domain part) exceeds a specified threshold.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()  # Extract domain from URL and convert to lowercase

    # Calculate the length of the domain (hostname)
    hostname_length = len(domain)

    # Compare the hostname length with the specified threshold
    if hostname_length > threshold:
        return 1
    else:
        return 0
    


def count_punctuation_symbols(url, punctuation_symbols):
    """
    Count the occurrences of specified punctuation symbols in the URL path and query parameters.
    """
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()  # Extract path from URL and convert to lowercase
    query = parsed_url.query.lower()  # Extract query parameters from URL and convert to lowercase

    # Combine path and query for comprehensive analysis
    url_text = path + '?' + query

    # Initialize a dictionary to store counts of punctuation symbols
    symbol_counts = {symbol: 0 for symbol in punctuation_symbols}

    # Count occurrences of punctuation symbols
    for char in url_text:
        if char in punctuation_symbols:
            symbol_counts[char] += 1

    total_count = sum(symbol_counts.values())

    return total_count


def check_redirect(url):
    status_code=0
    redirect=False
    try:
        response = requests.head(url, timeout=5)  # Set timeout to 5 seconds
        status_code=response.status_code
        if response.status_code not in [301, 302, 307, 308]:
            redirect=False
        else:
            redirect=True
        return [redirect,status_code]
    except requests.exceptions.RequestException as e:
        return [redirect,status_code]
    

def count_colons_in_hostname(url):
    """
    Count the occurrences of ':' in the URL's hostname (domain part).
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()  # Extract domain from URL and convert to lowercase

    # Count the number of colons (':') in the domain (hostname)
    colon_count = domain.count(':')
    
    return colon_count

def has_ip_address_or_hexadecimal(url):
    """
    Check if the URL contains an IP address or hexadecimal representation in the hostname.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()  # Extract domain from URL and convert to lowercase

    # Check for IP address pattern (e.g., '120.10.10.8')
    ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
    if ip_pattern.match(domain):
        return 1

    # Check for hexadecimal representation pattern (e.g., '0x78.0xA.0xA.8')
    hex_pattern = re.compile(r'^0[xX][0-9a-fA-F]+\.[0-9a-fA-F]+\.[0-9a-fA-F]+\.[0-9a-fA-F]+$')
    if hex_pattern.match(domain):
        return 1

    return 0

def calculate_vowel_consonant_ratio(url):
    """
    Calculate the ratio of total vowels to total consonants in the URL's hostname (domain part).
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()  # Extract domain from URL and convert to lowercase

    vowels = set('aeiou')  # Define set of vowel characters
    vowel_count = sum(1 for char in domain if char in vowels)  # Count vowels
    consonant_count = sum(1 for char in domain if char.isalpha() and char not in vowels)  # Count consonants

    # Calculate the ratio of vowels to consonants (avoid division by zero)
    if consonant_count > 0:
        ratio = vowel_count / consonant_count
    else:
        ratio = 0
    return ratio

def is_short_hostname_phishy(url, threshold=5):
    """
    Check if the length of the URL's hostname (domain part) is shorter than a specified threshold.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()  # Extract domain from URL and convert to lowercase

    # Calculate the length of the domain (hostname)
    hostname_length = len(domain)

    # Check if the hostname length is shorter than the threshold
    if hostname_length < threshold:
        return 1
    else:
        return 0
    
def has_at_symbol(url):
    """
    Check if the URL contains the '@' symbol, which can be used to deceive users.
    """
    parsed_url = urlparse(url)

    # Extract netloc (domain part) from the URL
    netloc = parsed_url.netloc.lower()

    # Check if the '@' symbol exists in the netloc (domain part)
    if '@' in netloc:
        return 1
    else:
        return 0

def process_url(row, features, lock):
    url = row['url']
    url_type = row['type']
    ip_address=row['ip_address']
    nameservers=row['nameservers']

    kl_div = frequncies_for_KL(url)
    entropy = calculate_entropy(url)
    digit_letter_ratio = calculate_digit_letter_ratio(url)
    top_level_domains_count = count_top_level_domains_in_path(url)
    dash_count = count_dashes_in_path(url)
    url_length = urllength(url)
    digits_in_domain = has_digits_in_domain(url)
    suspicious_words_count = count_suspicious_words(url, suspicious_words)
    subdomains_count = count_subdomains(url)
    brand_name_modified = is_brand_name_modified(url, popular_brands)
    long_hostname_phishy = is_long_hostname_phishy(url)
    punctuation_symbols_count = count_punctuation_symbols(url, punctuation_symbols)
    colons_in_hostname_count = count_colons_in_hostname(url)
    ip_address_or_hexadecimal = has_ip_address_or_hexadecimal(url)
    vowel_consonant_ratio = calculate_vowel_consonant_ratio(url)
    short_hostname_phishy = is_short_hostname_phishy(url)
    at_symbol = has_at_symbol(url)

    feature_values = [
        url,
        url_type,
        ip_address,
        nameservers,
        kl_div,
        entropy,
        digit_letter_ratio,
        top_level_domains_count,
        dash_count,
        url_length,
        digits_in_domain,
        suspicious_words_count,
        subdomains_count,
        brand_name_modified,
        long_hostname_phishy,
        punctuation_symbols_count,
        colons_in_hostname_count,
        ip_address_or_hexadecimal,
        vowel_consonant_ratio,
        short_hostname_phishy,
        at_symbol
    ]

    lock.acquire()
    features.append(feature_values)
    lock.release()

def main():
    urls_file = "processed_URLs.csv"
    df = pd.read_csv(urls_file)
    #Sampling is for testing purposes to check the model performance for earlier results on a samll dataset
    df_sample = df.sample(n=100000, random_state=42)

    features = []
    lock = threading.Lock()

    threads = []
    for index, row in df.iterrows():
        thread = threading.Thread(target=process_url, args=(row, features, lock))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    columns = [
    'url', 'type', 'ip_address', 'nameservers', 'kl_divergence', 'entropy', 'digit_letter_ratio', 
    'top_level_domains_count', 'dash_count', 'url_length', 'digits_in_domain', 'suspicious_words_count', 
    'subdomains_count', 'brand_name_modified', 'long_hostname_phishy', 'punctuation_symbols_count', 
    'colons_in_hostname_count', 'ip_address_or_hexadecimal', 'vowel_consonant_ratio', 'short_hostname_phishy', 
    'at_symbol'
    ]
    features_df = pd.DataFrame(features, columns=columns)

    features_df.to_csv("extracted_features.csv", index=False,mode='w')

    print(features_df.head())

if __name__ == "__main__":
    main()