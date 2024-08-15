def call_gpt4o(api_url, auth_token, app_name, question, image_url, timeout=30):
    headers = {
        'Content-Type': 'application/json',
        'authorization': auth_token
    }

    data = {
        "name": app_name,
        "inputs": {
            "stream": False,
            "msg": question,
            "image_url": image_url
        }
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(data), timeout=timeout)
        print(f"Status Code: {response.status_code}")
        # print(f"Headers: {response.headers}")
        # print(f"Content: {response.content}")

        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()
    except requests.exceptions.Timeout:
        print("Request timed out")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        raise

def get_response_gpt4o(prompt, image_url):
    api_url = 'https://llmaiadmin-test.classba.cn/api/chat/call'
    auth_token = 'ab3ed703b680e2b1a3845df109143879'  # 实际授权令牌
    app_name = 'yibo_test' # 这个可以在松鼠中台配置新的应用，并命名
    for _ in range(3):
        try:
            result = call_gpt4o(api_url, auth_token, app_name, prompt, image_url)
            print("API Response:\n", result['data'])
            print("#"*20)
            return str(result['data'])
        except:
            continue
    return ""