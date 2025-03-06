import pandas as pd
import numpy as np
import requests
import os
import time
from datetime import datetime, timedelta
import math
from bs4 import BeautifulSoup
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

# Base weights for different news types
NEWS_WEIGHTS = {
    "war": 0.9,
    "conflict": 0.7,
    "military": 0.6,
    "inflation": 0.8,
    "interest_rate": 0.7,
    "fed": 0.7,
    "central_bank": 0.6,
    "usd": 0.7,
    "dollar": 0.7,
    "gold": 0.9,
    "xau": 0.9,
    "precious_metal": 0.7,
    "economic_crisis": 0.8,
    "financial_crisis": 0.8,
    "recession": 0.8
}

# Urgency keywords and their multipliers
URGENCY_KEYWORDS = {
    "crisis": 1.5,
    "emergency": 1.7,
    "urgent": 1.5,
    "breaking": 1.3,
    "alert": 1.3,
    "nuclear": 2.0,
    "threat": 1.4,
    "war": 1.5,
    "attack": 1.4,
    "invasion": 1.5,
    "collapse": 1.6,
    "crash": 1.6,
    "surge": 1.3,
    "plunge": 1.4,
    "soar": 1.3,
    "dramatic": 1.2,
    "unexpected": 1.2,
    "surprise": 1.2,
    "shock": 1.4,
    "immediate": 1.3
}

# Time decay parameter (lambda)
DECAY_LAMBDA = 0.01  # Adjust this value to control decay rate

class NewsDataCollector:
    def __init__(self, save_dir="Data"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Create a dataframe to store all news data
        self.news_data = pd.DataFrame(columns=[
            'timestamp', 'datetime', 'source', 'title', 'content', 'url', 
            'news_type', 'sentiment_score', 'sentiment_label', 'weight', 
            'time_decay_factor', 'alt_decay_factor', 'urgency_label', 
            'adjusted_weight', 'alt_adjusted_weight'
        ])
    
    def calculate_time_decay(self, news_time):
        """Calculate time decay factor based on hours since the event"""
        current_time = datetime.now()
        if isinstance(news_time, str):
            try:
                news_time = datetime.strptime(news_time, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                try:
                    news_time = datetime.strptime(news_time, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    try:
                        news_time = datetime.strptime(news_time, "%B %d, %Y")
                    except ValueError:
                        # If parsing fails, use a default time (1 day ago)
                        news_time = current_time - timedelta(days=1)
        
        hours_diff = (current_time - news_time).total_seconds() / 3600
        decay_factor = math.exp(-DECAY_LAMBDA * hours_diff)
        return decay_factor, hours_diff
    
    def alternative_time_decay(self, base_weight, hours_since_event, decay_rate=0.01):
        """
        Alternative time decay function using a different formula
        This provides a more gradual decay compared to exponential
        """
        return base_weight * (1 / (1 + decay_rate * hours_since_event))
    
    def get_sentiment_score(self, text):
        """Get sentiment score using VADER sentiment analyzer"""
        if not text or len(text) == 0:
            return 0.0, "neutral"
        
        try:
            # Get sentiment scores
            sentiment = self.sentiment_analyzer.polarity_scores(text)
            
            # Convert compound score to range from -1 to 1
            sentiment_score = sentiment['compound']
            
            # Determine sentiment label
            if sentiment_score >= 0.05:
                sentiment_label = "positive"
            elif sentiment_score <= -0.05:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
                
            return sentiment_score, sentiment_label
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return 0.0, "neutral"
    
    def detect_news_type(self, text):
        """Detect the type of news based on keywords"""
        text = text.lower()
        detected_types = []
        
        for news_type in NEWS_WEIGHTS.keys():
            if news_type in text:
                detected_types.append(news_type)
        
        if not detected_types:
            return "general"
        
        return ", ".join(detected_types)
    
    def detect_urgency(self, text):
        """Detect urgency level based on keywords and return multiplier"""
        text = text.lower()
        urgency_multiplier = 1.0
        detected_urgency_keywords = []
        
        for keyword, multiplier in URGENCY_KEYWORDS.items():
            if keyword in text:
                urgency_multiplier = max(urgency_multiplier, multiplier)
                detected_urgency_keywords.append(keyword)
        
        urgency_label = ", ".join(detected_urgency_keywords) if detected_urgency_keywords else "normal"
        return urgency_multiplier, urgency_label
    
    def calculate_base_weight(self, news_type):
        """Calculate base weight based on news type"""
        if news_type == "general":
            return 0.3
        
        types = news_type.split(", ")
        weight = 0.0
        for t in types:
            if t in NEWS_WEIGHTS:
                weight = max(weight, NEWS_WEIGHTS[t])
        
        return weight if weight > 0.0 else 0.3
    
    def scrape_wsj(self):
        """Scrape news from Wall Street Journal with focus on specific keywords"""
        urls = [
            "https://www.wsj.com/news/markets?mod=nav_top_section",
            "https://www.wsj.com/news/economy?mod=nav_top_section",
            "https://www.wsj.com/news/business?mod=nav_top_section",
            "https://www.wsj.com/market-data/commodities?mod=nav_top_section",
            "https://www.wsj.com/search?query=gold&isToggleOn=true&operator=OR&sort=date-desc&duration=1d&startDate=2024%2F01%2F01&endDate=2024%2F12%2F31",
            "https://www.wsj.com/search?query=inflation&isToggleOn=true&operator=OR&sort=date-desc&duration=1d&startDate=2024%2F01%2F01&endDate=2024%2F12%2F31",
            "https://www.wsj.com/search?query=war&isToggleOn=true&operator=OR&sort=date-desc&duration=1d&startDate=2024%2F01%2F01&endDate=2024%2F12%2F31",
            "https://www.wsj.com/search?query=usd&isToggleOn=true&operator=OR&sort=date-desc&duration=1d&startDate=2024%2F01%2F01&endDate=2024%2F12%2F31",
            "https://www.wsj.com/search?query=recession&isToggleOn=true&operator=OR&sort=date-desc&duration=1d&startDate=2024%2F01%2F01&endDate=2024%2F12%2F31"
        ]
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        }
        
        for url in urls:
            try:
                print(f"Scraping news from {url}...")
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Updated selectors for WSJ's current structure
                articles = soup.select("article, div.WSJTheme--story--XB4V2mLz, div.article-wrap, div.WSJTheme--headline--7VCzo7Ay, div.WSJBase--card-container--1TwzYqGz")
                
                print(f"Found {len(articles)} potential articles")
                
                for article in articles:
                    try:
                        # Updated title and link selectors
                        title_elem = article.select_one("h2 a, h3 a, a.WSJTheme--headline-link--3qHx_o9G, div.WSJTheme--headline--7VCzo7Ay a, a[data-reflink='article_link']")
                        if not title_elem:
                            continue
                        title = title_elem.text.strip()
                        link = title_elem['href'] if title_elem.has_attr('href') else ""
                        
                        # Add more debug information
                        print(f"Found article: {title}")
                        print(f"Link: {link}")
                        
                        # Check if the article contains any of our target keywords
                        title_lower = title.lower()
                        if not any(keyword in title_lower for keyword in ["war", "usd", "gold", "inflation", "recession", "dollar", "fed", "crisis", "economy", "market"]):
                            # Skip articles that don't match our focus keywords
                            print(f"Skipping article as it doesn't match keywords: {title}")
                            continue
                        
                        # Extract date
                        date_elem = article.select_one("div.WSJTheme--timestamp--22sfkNDv, span.date-line, time, span.timestamp, p.timestamp")
                        date_str = date_elem.text.strip() if date_elem else ""
                        
                        # Parse date
                        dt = None
                        if date_str:
                            try:
                                # Try to parse date formats like "Jan. 1, 2023"
                                dt = datetime.strptime(date_str, "%b. %d, %Y")
                            except:
                                try:
                                    # Try another format
                                    dt = datetime.strptime(date_str, "%B %d, %Y")
                                except:
                                    try:
                                        # Try format with time
                                        dt = datetime.strptime(date_str, "%b %d, %Y %I:%M %p ET")
                                    except:
                                        dt = datetime.now()
                        else:
                            dt = datetime.now()
                        
                        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Get content if available
                        content = ""
                        if link and not link.startswith("http"):
                            link = "https://www.wsj.com" + link
                            
                        if link:
                            try:
                                # Wait to avoid rate limiting
                                time.sleep(1)
                                
                                article_response = requests.get(link, headers=headers)
                                article_soup = BeautifulSoup(article_response.text, 'html.parser')
                                
                                # Extract article content - WSJ has different content structures
                                content_elems = article_soup.select("div.article-content p, div.WSJTheme--body--1qrvzYXI p, div.article__body p")
                                if content_elems:
                                    content = " ".join([p.text.strip() for p in content_elems])
                                print(f"Got content: {content[:100]}...")
                            except Exception as e:
                                print(f"Error fetching article content: {e}")
                        
                        # Process the article
                        self._process_article(timestamp_str, dt, "WSJ", title, content, link)
                        print(f"Processed article: {title}")
                        
                    except Exception as e:
                        print(f"Error processing article: {e}")
                
                print(f"Processed {len(articles)} articles from {url}")
                
                # Wait between requests to avoid rate limiting
                time.sleep(2)
                
            except Exception as e:
                print(f"Error scraping {url}: {e}")
    
    def scrape_kitco(self):
        """Scrape news from Kitco (gold-specific news)"""
        url = "https://www.kitco.com/news/gold/"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        try:
            print(f"Scraping news from {url}...")
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news articles
            articles = soup.select("div.article-list-item")
            
            for article in articles:
                try:
                    # Extract title and link
                    title_elem = article.select_one("h3.article-list-item__title a")
                    if not title_elem:
                        continue
                        
                    title = title_elem.text.strip()
                    link = title_elem['href'] if title_elem.has_attr('href') else ""
                    
                    # Extract date
                    date_elem = article.select_one("div.article-list-item__date")
                    date_str = date_elem.text.strip() if date_elem else ""
                    
                    # Parse date
                    dt = None
                    if date_str:
                        try:
                            dt = datetime.strptime(date_str, "%B %d, %Y")
                        except:
                            dt = datetime.now()
                    else:
                        dt = datetime.now()
                    
                    timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Get content if available
                    content = ""
                    if link:
                        try:
                            # Wait to avoid rate limiting
                            time.sleep(1)
                            
                            article_response = requests.get(link, headers=headers)
                            article_soup = BeautifulSoup(article_response.text, 'html.parser')
                            
                            # Extract article content
                            content_elem = article_soup.select_one("div.article__content")
                            if content_elem:
                                content = content_elem.text.strip()
                        except Exception as e:
                            print(f"Error fetching article content: {e}")
                    
                    # Process the article
                    self._process_article(timestamp_str, dt, "Kitco", title, content, link)
                    
                except Exception as e:
                    print(f"Error processing article: {e}")
            
            print(f"Processed {len(articles)} articles from {url}")
            
            # Wait between requests to avoid rate limiting
            time.sleep(2)
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    
    def _process_article(self, timestamp_str, dt, source, title, content, url):
        """Process a news article and add it to the dataframe"""
        # Combine title and content for analysis
        full_text = f"{title} {content}"
        
        # Get sentiment score
        sentiment_score, sentiment_label = self.get_sentiment_score(full_text)
        
        # Detect news type
        news_type = self.detect_news_type(full_text)
        
        # Calculate base weight
        base_weight = self.calculate_base_weight(news_type)
        
        # Calculate time decay
        decay_factor, hours_diff = self.calculate_time_decay(dt)
        
        # Calculate alternative time decay
        alt_decay_factor = self.alternative_time_decay(base_weight, hours_diff)
        
        # Detect urgency
        urgency_multiplier, urgency_label = self.detect_urgency(full_text)
        
        # Calculate adjusted weight (using original decay factor)
        adjusted_weight = base_weight * decay_factor * urgency_multiplier
        
        # Calculate alternative adjusted weight
        alt_adjusted_weight = alt_decay_factor * urgency_multiplier
        
        # Add to dataframe
        new_row = pd.DataFrame([{
            'timestamp': int(dt.timestamp()) if isinstance(dt, datetime) else 0,
            'datetime': timestamp_str,
            'source': source,
            'title': title,
            'content': content[:500] + "..." if len(content) > 500 else content,  # Truncate long content
            'url': url,
            'news_type': news_type,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'weight': base_weight,
            'time_decay_factor': decay_factor,
            'alt_decay_factor': alt_decay_factor / base_weight,  # Normalize to compare with original
            'urgency_label': urgency_label,
            'adjusted_weight': adjusted_weight,
            'alt_adjusted_weight': alt_adjusted_weight
        }])
        
        self.news_data = pd.concat([self.news_data, new_row], ignore_index=True)
    
    def save_data(self):
        """Save collected news data to CSV"""
        if len(self.news_data) > 0:
            # Sort by adjusted weight (descending) and timestamp (descending)
            self.news_data = self.news_data.sort_values(by=['adjusted_weight', 'timestamp'], ascending=[False, False])
            
            # Save to CSV
            csv_path = os.path.join(self.save_dir, "news_data.csv")
            self.news_data.to_csv(csv_path, index=False)
            print(f"Saved {len(self.news_data)} news items to {csv_path}")
            return csv_path
        else:
            print("No news data to save")
            return None
    
    def visualize_sentiment(self):
        """Create visualizations of news sentiment"""
        if len(self.news_data) == 0:
            print("No data to visualize")
            return
        
        # Create a directory for visualizations
        viz_dir = os.path.join(self.save_dir, "Visualizations")
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # 1. Sentiment distribution by news source
        plt.figure(figsize=(12, 6))
        source_sentiment = self.news_data.groupby('source')['sentiment_score'].mean().sort_values()
        source_sentiment.plot(kind='barh', color=plt.cm.RdYlGn(
            (source_sentiment + 1) / 2))  # Map -1,1 to 0,1 for color scale
        plt.title('Average Sentiment Score by News Source')
        plt.xlabel('Sentiment Score (-1: Negative, 1: Positive)')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'sentiment_by_source.png'))
        
        # 2. Sentiment over time
        plt.figure(figsize=(14, 7))
        # Convert timestamp to datetime for plotting
        self.news_data['plot_date'] = pd.to_datetime(self.news_data['datetime'])
        # Sort by date
        time_data = self.news_data.sort_values('plot_date')
        # Plot sentiment scores
        plt.scatter(time_data['plot_date'], time_data['sentiment_score'], 
                    c=time_data['sentiment_score'], cmap='RdYlGn', 
                    alpha=0.7, s=time_data['adjusted_weight']*100)
        plt.colorbar(label='Sentiment Score')
        plt.title('News Sentiment Over Time (Size indicates importance)')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'sentiment_over_time.png'))
        
        # 3. News type distribution
        plt.figure(figsize=(12, 8))
        # Extract all news types (can be multiple per article)
        all_types = []
        for types in self.news_data['news_type']:
            if types != 'general':
                all_types.extend(types.split(', '))
        
        type_counts = pd.Series(all_types).value_counts()
        type_counts.plot(kind='barh', color='skyblue')
        plt.title('Distribution of News Types')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'news_type_distribution.png'))
        
        print(f"Visualizations saved to {viz_dir}")

# Add a proper main function to run the scraper
def main():
    # Create news collector
    collector = NewsDataCollector(save_dir="Data")
    
    try:
        # Scrape news from WSJ
        print("Starting to scrape WSJ...")
        collector.scrape_wsj()
        
        # Scrape news from Kitco
        print("Starting to scrape Kitco...")
        collector.scrape_kitco()
        
        # Save collected data
        print("Saving data...")
        csv_path = collector.save_data()
        
        # Create visualizations
        collector.visualize_sentiment()
        
        print("\nNews data collection completed!")
        print(f"Total articles collected: {len(collector.news_data)}")
        
    except Exception as e:
        print(f"An error occurred during news collection: {e}")
        import traceback
        traceback.print_exc()  # Print the full error traceback for debugging

# Call the main function when the script is run directly
if __name__ == "__main__":
    main()