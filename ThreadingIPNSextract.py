import pandas as pd
import dns.resolver
import csv
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

def extract_ip_and_nameservers(domain):
    ip_address = []
    nameservers = []

    try:
        if is_ip_address(domain):  # Check if the input is an IP address
            # Perform reverse DNS lookup to get the associated domain name
            ip_address = [domain]

        else:    
            # Resolve A (IPv4) records for the domain
            answers = dns.resolver.resolve(domain, 'A')
            ip_address = [str(rdata) for rdata in answers]

        # Resolve NS (nameserver) records for the domain
        answers = dns.resolver.resolve(domain, 'NS')
        nameservers = [str(rdata) for rdata in answers]

    except dns.resolver.NoAnswer:
        print(f"No DNS records found for domain '{domain}'")
    except dns.resolver.NXDOMAIN:
        print(f"Domain '{domain}' does not exist")

    return ip_address, nameservers

def is_ip_address(domain):
    """Check if the input string is an IPv4 address."""
    parts = domain.split('.')
    if len(parts) == 4:
        return all(part.isdigit() and 0 <= int(part) <= 255 for part in parts)
    return False

def process_url(url, df):
    processed_data = {}

    try:
        parsed_url = urlparse(url)

        if parsed_url.netloc:  # Check if netloc (domain) is present
            domain = parsed_url.netloc
        else:
            # Extract domain from the path (e.g., for relative URLs)
            domain = parsed_url.path.split('/')[0]

        if domain:
            # Extract IP address and nameservers for the domain
            ip_address, nameservers = extract_ip_and_nameservers(domain)

            # Get the type of the URL from the DataFrame
            url_type = df.loc[df['url'] == url, 'type'].values[0]

            # Prepare processed data for the URL
            processed_data = {
                'url': url,
                'type': url_type,
                'ip_address': ', '.join(ip_address),
                'nameservers': ', '.join(nameservers)
            }
        else:
            print(f"Invalid URL: '{url}'")
    except Exception as e:
        print(f"Error processing URL '{url}': {e}")

    return processed_data

def process_urls(input_file, output_file):
    # Read CSV file containing URLs and types
    df = pd.read_csv(input_file)

    # List of URLs to process
    urls = df['url'].tolist()

    processed_data = []
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(process_url, urls, [df]*len(urls))

        for result in results:
            if result:  # Check if result is not empty
                processed_data.append(result)

    # Write processed data to a new CSV file
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['url', 'type', 'ip_address', 'nameservers']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(processed_data)

    print(f"Processed data saved to '{output_file}'")

if __name__ == "__main__":
    input_csv_file = 'malicious_phish.csv'
    output_csv_file = 'processed_URLs.csv'

    process_urls(input_csv_file, output_csv_file)
