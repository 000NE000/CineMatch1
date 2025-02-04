# Store raw data as JSON initially

from logs import setup_logging
from input import load_movies
from scraper import scrape_reviews
from storage import save_to_json, save_to_csv
import logging

def main():
    # 단계 1: 로깅 설정
    setup_logging()
    
    # 단계 2: 입력 데이터 로드
    movies = load_movies('movie_brief_info.json')
    if not movies:
        logging.error("스크래핑할 영화가 없습니다. 종료합니다.")
        return
    
    # 단계 3: 각 영화에 대한 리뷰 스크래핑
    enriched_data = []
    for movie in movies:
        title = movie.get('title', 'Unknown Title')
        url = movie.get('url', '')
        if not url:
            logging.warning(f"영화 '{title}'에 URL이 없습니다. 스킵합니다.")
            continue
        logging.info(f"'{title}'의 리뷰 스크래핑을 시작합니다.")
        reviews = scrape_reviews(url, max_pages=2)
        movie['reviews'] = reviews
        enriched_data.append(movie)
        logging.info(f"'{title}'의 스크래핑을 완료했습니다. 총 {len(reviews)}개의 리뷰가 수집되었습니다.")
    
    # 단계 4: 스크래핑된 데이터 저장
    save_to_json(enriched_data, 'data/movies_with_reviews.json')
    save_to_csv(enriched_data, 'data/reviews.csv')
    logging.info("모든 데이터가 성공적으로 저장되었습니다.")

if __name__ == '__main__':
    main()