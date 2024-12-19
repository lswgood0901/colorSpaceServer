from flask import Flask, request, jsonify
from optimizer import BayesianOptimizer
from openai import OpenAI
import json

client = OpenAI(api_key = "your api")
app = Flask(__name__)

# OpenAI API 호출 함수
def call_openai_api(system_prompt, user_prompt):
    try:
        # Chat Completion 호출
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",  # 모델 이름
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        return completion.choices[0].message.content.strip().split("\n")
    
    except Exception as e:
        return {"error": f"Error: {str(e)}"}  # 에러 발생 시 딕셔너리 형태로 반환
    
optimizer = BayesianOptimizer(dimension=3)

@app.route('/observe_user_behavior', methods=['POST'])
def observe_user_behavior():
    global optimizer

    data = request.json
    chosen_data = data.get('chosen', [])
    other_data = data.get('others', [])

    # 정규화 함수 정의
    def normalize_rgb(rgb):
        return [x / 255.0 for x in rgb]

    # 입력 데이터 확인
    if not chosen_data or not other_data:
        return jsonify({'error': 'Invalid input data'}), 400

    print("Original chosen data:", chosen_data)
    print("Original other data:", other_data)

    # 정규화된 데이터 준비
    normalized_chosen = normalize_rgb(chosen_data)
    normalized_others = [normalize_rgb(rgb) for rgb in other_data]

    print("Normalized chosen data:", normalized_chosen)
    print("Normalized other data:", normalized_others)

    # Optimizer에 데이터 전달
    optimizer.observe_behaivor_estimate(normalized_chosen, normalized_others)

    # 새로운 후보 생성
    new_candidates = optimizer.optimize_acqf_and_get_observation().tolist()
    int_candidates = [
        [int(value * 255) for value in candidate] for candidate in new_candidates
    ]
    response_data = {
            "suggested_rgb": int_candidates
        }
    return jsonify(response_data), 200


@app.route('/upload_text', methods=['POST'])
def suggest_colors_gpt():
    global optimizer, client
    try:
        data = request.json
        user_text = data.get("text")  # 사용자의 텍스트 입력
        current_rgb = data.get("current_rgb")  # 현재 RGB 값 (리스트 형태: [R, G, B])
        print("data",data)
        # if not user_text or not current_rgb:
        #     return jsonify({"error": "Invalid input data"}), 400
        # ChatGPT API 요청
        print("Preparing to call OpenAI API...")
        system_prompt = "You are an assistant specialized in color processing."
        user_prompt = f"""
        You are an AI assistant. The user provided a text input and a current RGB value. 
        Your job is to:
        1. Understand the user's intention, even if the text has typos or is unclear.
        2. If the user's input is **specific** (e.g., "Make it red"), provide three RGB values within the same color family but with noticeable differences in shade, saturation, or brightness.
        3. If the user's input is **abstract** (e.g., "Make it more attractive"), provide three RGB values that explore different color families and are significantly distinct from each other.
        4. Consider the user's current RGB value when suggesting new RGB values, ensuring the suggestions align with the user's context or contrast appropriately.
        5. Respond in JSON format with the following fields:
        - "recognized_text": The corrected or interpreted text of the user's input.
        - "suggested_rgb": A list of three RGB values, each as an array of three integers (e.g., [255, 0, 0]).
        - "notes": A brief explanation of each suggestion, including:
            * How the suggestion relates to the user's input.
            * For abstract inputs, why different color families were chosen.
            * For specific inputs, what variations within the color family were made and why.
        
        User Input Text: {user_text}
        Current RGB: {current_rgb}
        """
        # user_prompt = "I want more pink color of this chair."
        response = call_openai_api(system_prompt, user_prompt)

        # OpenAI API 응답 디버깅
        print("OpenAI API raw response:", response)
        
        # JSON 데이터 추출
        json_data_str = "\n".join(response[1:-1])  # "```json"과 "```" 제거
        print("Extracted JSON String:", json_data_str)

        # 문자열을 JSON으로 파싱
        parsed_json = json.loads(json_data_str)
        print("Parsed JSON:", parsed_json)

        # 'suggested_rgb' 값 추출
        suggested_rgb = parsed_json.get("suggested_rgb", None)
        print("Suggested RGB:", suggested_rgb)
        response_data = {
            "suggested_rgb": suggested_rgb
        }
        # 응답 반환
        return jsonify(response_data), 200


    except Exception as e:
        print("Error occurred:", str(e))  # 디버깅: 발생한 에러 출력
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
