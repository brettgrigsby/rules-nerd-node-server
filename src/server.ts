import express from "express"
import cors from "cors"
import bodyParser from "body-parser"
import dotenv from "dotenv"
import { OpenAIEmbeddings } from "langchain/embeddings/openai"
import { VectorDBQAChain } from "langchain/chains"
import { PineconeStore } from "langchain/vectorstores/pinecone"
import { PineconeClient } from "@pinecone-database/pinecone"
import { OpenAI } from "langchain/llms/openai"

dotenv.config()
const client = new PineconeClient()

const app = express()

app.use(bodyParser.json())
app.use(cors())

let qaChain: VectorDBQAChain

const initializeData = async () => {
  try {
    console.log("Initializing data")
    // const fileContents = fs.readFileSync(
    //   "src/rules-docs/magic-the-gathering.txt",
    //   "utf8"
    // )
    // const docs = fileContents.split("\r\r").filter(Boolean)
    // const docs = await textSplitter.splitText(fileContents)
    // docs = await mySplitter.createDocuments([fileContents])

    const openAI = new OpenAI({
      openAIApiKey: process.env.OPENAI_API_KEY,
    })

    await client.init({
      apiKey: process.env.PINECONE_API_KEY,
      environment: process.env.PINECONE_ENVIRONMENT,
    })
    const pineconeIndex = client.Index(process.env.PINECONE_INDEX)

    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    })

    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex,
    })

    // const docSearch = await MemoryVectorStore.fromTexts(
    //   docs,
    //   docs.map((d, i) => ({ id: i, game: "magic-the-gathering" })),
    //   embeddings
    // )

    qaChain = VectorDBQAChain.fromLLM(openAI, vectorStore, {
      returnSourceDocuments: true,
    })
    console.log("Initialization complete")
  } catch (e) {
    console.log("Error initializing data")
    console.log(e)
  }
}

initializeData()

app.get("/", async (req, res) => {
  res.json({ status: 200 }).status(200)
})

app.post("/query", async (req, res) => {
  const { query } = req.body
  console.log({ query })
  const result = await qaChain.call({ query })
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
