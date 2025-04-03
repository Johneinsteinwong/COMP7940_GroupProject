import logging

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_openai.llms.base import BaseOpenAI

from langchain_core.globals import set_verbose, set_debug
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import Redis, Chroma
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import configparser
import re, os, json, tweepy
#import redis
import psycopg2
from datetime import datetime, timezone, timedelta
from ChatGPT_HKBU import HKBU_ChatGPT


set_debug(True)
set_verbose(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self, config: configparser.ConfigParser, embedding_model: str = "mxbai-embed-large"):
        self.model = HKBU_ChatGPT(
            base_url=config['CHATGPT']['BASICURL'],
            model=config['CHATGPT']['MODELNAME'], 
            api_version=config['CHATGPT']['APIVERSION'],
            api_key=config['CHATGPT']['ACCESS_TOKEN'],
        )
        self.config = config
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        '''
        self.embeddings = OpenAIEmbeddings( 
            base_url=config['CHATGPT']['BASICURL'],
            model=config['CHATGPT']['EMBEDDING'],
            api_version=config['CHATGPT']['APIVERSION'],
            api_key=config['CHATGPT']['ACCESS_TOKEN'],
        )'''

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant answering questions based on the uploaded document within 3 to 4 sentences.
            Context:
            {context}
            
            Question:
            {question}
            
           
            """
        )
        self.vector_store = PGVector.from_existing_index(
            embedding=self.embeddings,
            connection=self.config['PostgreSQL']['CONNECTION_STRING'],
            collection_name=self.config['PostgreSQL']['INDEX_NAME'],
        )
        #self.vector_store = Redis(
        #    embedding=self.embeddings,
        #    redis_url=self.config['REDIS']['HOST'],
        #    index_name=self.config['REDIS']['INDEX_NAME'],
        #)
        self.retriever = None


    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Ask a question based on the ingested document and return the answer.
        """
        if self.vector_store is None:
            raise ValueError("No document has been ingested yet.")

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
            )

        logging.info(f"Retrieving context for query: {query}")
        context = self.retriever.invoke(query)
        logging.info(f"Retrieved context: {context}")

        if not context:
            return "No relevant context found."
        
        formatted_prompt = {
            "context": "\n\n".join(doc.page_content for doc in context),
            "question": query,
        }

        chain = (
            RunnablePassthrough() 
            | self.prompt         
            | self.model           
            | StrOutputParser()    
        )

        logger.info("Generating response using the LLM.")
        reply = chain.invoke(formatted_prompt)
        reply = re.sub(r'<think>.*?</think>\n?\s*', '', reply, flags=re.DOTALL).strip()
        return reply
    
    def summarize_tweets(self, tweets: str):

        prompt = ChatPromptTemplate.from_template(
            """
            You are a given tweets with timestamps. 
            Tweets:
            {context}
            Question:
            {question}
            """
        )

        context = {
            "context": tweets,
            "question": "Summarize the tweets in point form. Remove the hashtags, number the tweets and group them by date."
        }
        chain = (
            RunnablePassthrough() 
            | prompt         
            | self.model          
            | StrOutputParser()   
        )
        logger.info("Summarizing tweets.")
        reply = chain.invoke(context)
        reply = re.sub(r'<think>.*?</think>\n?\s*', '', reply, flags=re.DOTALL).strip()
        return reply



def summarize(update: Update, context: CallbackContext) -> None:
    try:
        #global redis1
        global postgreConn
        cur = postgreConn.cursor()
        start, end = context.args#[0]
        logging.info(context)

        start = datetime.fromisoformat(start)
        end = datetime.fromisoformat(end)
        
        num_days = (end - start).days
        if not (0 <= num_days <= 14):
            update.message.reply_text('Please enter a date range within 14 days.')
        else:
            #update.message.reply_text('Searching for the number of cases between ' + str(start.timestamp()) + ' and ' + str(end.timestamp()))
#
            start_timestamp = start#int(start.timestamp())
            end_timestamp = end + timedelta(days=1, seconds=-1) #int(end.timestamp()) + 86399 

            cur.execute("""
                SELECT id, created_at, text 
                FROM tweets 
                WHERE created_at BETWEEN %s AND %s
                ORDER BY created_at ASC
            """, (start_timestamp, end_timestamp))
            tweets = cur.fetchall()
            tweet_ids = [tweet[0] for tweet in tweets]

            # tweet_ids = redis1.zrangebyscore(
            #     "tweets:by_time",
            #     start_timestamp,
            #     end_timestamp,
            #     withscores=False  # Return only IDs (omit timestamps)
            # )
            logging.info(tweet_ids)
            #tweets_collected = []
            tweets_str = '='*50 + '\n'
            if tweet_ids:
                for id, created_at, text in tweets:
                    tweets_str += created_at.isoformat() + '\n'
                    tweets_str += text + '\n'
                    tweets_str += '='*50 + '\n'

                # for tweet_id in tweet_ids:

                #     tweet = redis1.hgetall(tweet_id)
                #     tweets_collected.append(tweet)
                #     tweets_str += datetime.fromtimestamp(int(tweet['created_at'])).isoformat()
                #     tweets_str += '\n'
                #     tweets_str += tweet['text'] + '\n'
                #     tweets_str += '='*50 + '\n'

                #tweets_json_str = json.dumps(tweets_collected, indent=2) 
                reply_message = chatbot.summarize_tweets(tweets_str)
                update.message.reply_text(reply_message)
            else:
                update.message.reply_text('No tweets found between ' + str(start) + ' and ' + str(end))

       # update.message.reply_text('You have said ' + msg + ' for ' + 
       #                         redis1.get(msg) + ' times.')
    except Exception as e:
        logging.error("Error in summarize: " + str(e))
        update.message.reply_text('Usage: /summarize <start date>(YYYY-MM-DD) <end date>(YYYY-MM-DD)')
    finally:
        cur.close()


def equiped_chatgpt(update: Update, context: CallbackContext) -> None:
    global chatbot
    reply_message = chatbot.ask(update.message.text)
    logging.info("Update: " + str(update))
    logging.info("Context: " + str(context))
    context.bot.send_message(chat_id=update.effective_chat.id, text=reply_message)

def create_table() -> None:
    global postgreConn
    cur = postgreConn.cursor()
    cur.execute(""" DROP TABLE IF EXISTS tweets """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tweets (
            id BIGINT PRIMARY KEY,
            created_at TIMESTAMP,
            text TEXT
        )
    """)
    postgreConn.commit()
    cur.close()

def check_tweet_exists(tweet_id):
    global postgreConn
    cur = postgreConn.cursor()
    cur.execute("""
        SELECT 1 FROM tweets WHERE id = %s LIMIT 1
    """, (tweet_id,))
    return cur.fetchone() is not None

def insert_data(BEARER_TOKEN) -> None:
    try:
        global postgreConn
        cur = postgreConn.cursor()
        # Get max timstamp
        cur.execute("""
            SELECT MAX(created_at) FROM tweets
        """)
        result = cur.fetchone()
        max_timestamp = result[0] if result[0] is not None else 0
        logging.info("Max timestamp: " + str(max_timestamp))
        endtime = datetime.now(timezone.utc)
        starttime = max_timestamp + timedelta(days=1) if max_timestamp else datetime.now(timezone.utc) - timedelta(days=100)
        

        client = tweepy.Client(bearer_token=BEARER_TOKEN)
        user_id = "1499972073728274433" # userid of PQabelian

        tweets_created_at = []
        for tweet in tweepy.Paginator(
            client.get_users_tweets,
            id=user_id,
            start_time=starttime,
            end_time=endtime,
            tweet_fields=["id","created_at", "text"],
            max_results=100
        ).flatten(limit=1000):
            if not check_tweet_exists(tweet.id):
                #tweets.append(tweet)
                cur.execute("""
                    INSERT INTO tweets (id, created_at, text) VALUES (%s, %s, %s)""",
                    (tweet["id"], tweet["created_at"], tweet["text"])
                )
                tweets_created_at.append(tweet["created_at"])
        postgreConn.commit()
        logging.info("Inserted " + str(len(tweets_created_at)) + " tweets into the database.")
        if tweet.created_at:
            logging.info("New max timestamp: " + max(tweets_created_at).isoformat())
    except Exception as e:
        logging.error("Error inserting data: " + str(e))
    finally:
        cur.close()

def main():
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    #os.environ['OPENAI_API_KEY'] = config['CHATGPT']['ACCESS_TOKEN']
    BEARER_TOKEN = config['TWITTER']['BEARER_TOKEN'] 

    updater = Updater(token=config['TELEGRAM']['ACCESS_TOKEN'], use_context=True)
    dispatcher = updater.dispatcher

    global postgreConn 
    postgreConn = psycopg2.connect(config['PostgreSQL']['CONNECTION_STRING'])

    create_table()
    insert_data(BEARER_TOKEN)
    
    # global redis1
    # redis1 = redis.Redis(
    #     host=config['REDIS']['HOST_RAW'],
    #     password=config['REDIS']['PASSWORD'],
    #     port=config['REDIS']['REDISPORT'],
    #     decode_responses=config['REDIS']['DECODE_RESPONSE'],
    #     username=config['REDIS']['USER_NAME']
    # )
    global chatbot
    chatbot = ChatBot(config)

    chatgpt_handler = MessageHandler(Filters.text & (~Filters.command), equiped_chatgpt)
    dispatcher.add_handler(chatgpt_handler)

    dispatcher.add_handler(CommandHandler('summarize',summarize))

    updater.start_polling()
    updater.idle()

    
if __name__ == "__main__":
    main()

