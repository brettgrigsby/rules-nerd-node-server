import fs from "fs"
import express from "express"
import cors from "cors"
import bodyParser from "body-parser"
import dotenv from "dotenv"
import { OpenAIEmbeddings } from "langchain/embeddings/openai"
import { RetrievalQAChain } from "langchain/chains"
import { MemoryVectorStore } from "langchain/vectorstores/memory"
import { OpenAI } from "langchain/llms/openai"

dotenv.config()

const app = express()

app.use(bodyParser.json())
app.use(cors())

let qa: RetrievalQAChain

const initializeData = async () => {
  console.log("Initializing data")
  console.log({ openAIApiKey: process.env.OPENAI_API_KEY })
  try {
    const openAI = new OpenAI({
      openAIApiKey: process.env.OPENAI_API_KEY,
    })
    const fileContents = fs.readFileSync(
      "src/rules-docs/magic-the-gathering.txt",
      "utf8"
    )
    console.log("got file contents")
    const docs = fileContents.split("\r\r").filter(Boolean)
    // const docs = await textSplitter.splitText(fileContents)
    // docs = await mySplitter.createDocuments([fileContents])
    console.log({ docsLen: docs.length })
    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    })
    console.log("got embeddings")
    const docSearch = await MemoryVectorStore.fromTexts(
      docs,
      docs.map((d, i) => ({ id: i, game: "magic-the-gathering" })),
      embeddings
    )
    console.log("got doc search")
    qa = RetrievalQAChain.fromLLM(openAI, docSearch.asRetriever(), {
      returnSourceDocuments: true,
    })
    console.log("defined chain")
    // const metaDatas = docs.map((doc, idx) => ({
    //   source: `${idx}-pl`,
    // }))
    // const docSearch = Chroma.fromTexts(docs, metaDatas, embeddings, {})
  } catch (e) {
    console.log("Error initializing data")
    console.log(e)
  }
}

initializeData()

app.get("/", async (req, res) => {
  const result = await qa.call({ query: "What is a creature?" })
  console.log({ result })
  res.json({ status: 200 }).status(200)
})

app.post("/query", async (req, res) => {
  const { query } = req.body
  console.log({ query })
  const result = await qa.call({ query })
  const { text, sourceDocuments } = result
  res
    .json({
      answer: text,
      sources: sourceDocuments.map((doc) => doc.pageContent),
    })
    .status(200)
})

const port = process.env.PORT || 4000
app.listen(port, () => {
  console.log(`Server running on port ${port}`)
})
