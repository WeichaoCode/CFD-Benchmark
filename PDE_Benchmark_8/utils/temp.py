import google.generativeai as genai

genai.configure(api_key="AIzaSyAlV3264W1Qoo_txpi7QCJz40PVN2jUhs4")
model = genai.GenerativeModel("models/gemini-1.5-flash")
prompt = "The quick brown fox jumps over the lazy dog."
# Call `count_tokens` to get the input token count (`total_tokens`).
print("total_tokens: ", model.count_tokens(prompt))
# ( total_tokens: 10 )
response = model.generate_content(prompt)
# On the response for `generate_content`, use `usage_metadata`
# to get separate input and output token counts
# (`prompt_token_count` and `candidates_token_count`, respectively),
# as well as the combined token count (`total_token_count`).
print(response.text)
# ( prompt_token_count: 11, candidates_token_count: 73, total_token_count: 84 )
