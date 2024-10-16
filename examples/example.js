const OpenAI = require('openai');

// укажите кредиты Yandex CLoud если используете проксю с включенной аутентификацией
// const FOLDER_ID = "";
// const API_KEY_OR_IAM_KEY = "";
// const key = `${FOLDER_ID}@${API_KEY_OR_IAM_KEY}`;

// или оставьте ключ sk-my, если создали проксю с авоматической аутентификацией 
const key = "sk-my";

// задайте адрес вашей прокси
const proxyUrl = "http://0.0.0.0:9041";

// создайте клиент OpenAI с измененным base_url
const openai = new OpenAI({
  apiKey: key,
  baseURL: `${proxyUrl}/v1/`,
});

async function generateTextOai(systemPrompt, userPrompt, maxTokens = 2000, temperature = 0.1, model = "yandexgpt/latest") {
  const response = await openai.chat.completions.create({
    messages: [
      {
        role: "system",
        content: systemPrompt,
      },
      {
        role: "user",
        content: userPrompt,
      },
    ],
    model: model,
    //max_tokens: maxTokens,
    //temperature: temperature,
  });

  const generatedText = response.choices[0].message.content;
  return generatedText;
}

async function getEmbedding(text, model = "text-search-doc/latest") {
    const response = await openai.embeddings.create({
      input: [text],
      model: model,
    });
    return response.data[0].embedding; // Возвращаем эмбеддинг
}

async function getEmbeddingSyncBatch(texts, model = "text-search-doc/latest") {
  const response = await openai.embeddings.create({
    input: texts,
    model: model,
  });
  return response.data; // Возвращаем эмбеддинг
}
  

async function main() {
  // Поддерживаемые форматы моделей
  const model = 'yandexgpt/latest'; 
  // или `gpt://${FOLDER_ID}/yandexgpt/latest` 
  // или `ds://${MODEL_ID}`
  // Для эмбеддингов 'text-search-doc/latest' 
  // или `emb://${FOLDER_ID}/text-search-doc/latest`
  // или `ds://${MODEL_ID}`

  const generatedText = await generateTextOai("You are a helpful assistant.", "What is the meaning of life? Answer in one word.");
  console.log(generatedText);

  const embedding = await getEmbedding("Hello Yandex!");
  console.log(embedding.slice(0, 3), '...');
}

main();
