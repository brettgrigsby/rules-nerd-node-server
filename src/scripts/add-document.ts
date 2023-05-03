import fs from "fs"
import path from "path"
import dotenv from "dotenv"
import pdfParse from "pdf-parse"
import { PineconeClient } from "@pinecone-database/pinecone"
import { Document } from "langchain/document"
import { PineconeStore } from "langchain/vectorstores/pinecone"
import { OpenAIEmbeddings } from "langchain/embeddings/openai"
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"

dotenv.config()

async function initializeIndex() {
  const client = new PineconeClient()
  await client.init({
    apiKey: process.env.PINECONE_API_KEY,
    environment: process.env.PINECONE_ENVIRONMENT,
  })
  return client.Index(process.env.PINECONE_INDEX)
}

// Function to read the contents of a file and log it to the terminal
function readFileContent(filePath: string): string {
  if (!fs.existsSync(filePath)) {
    console.error("File not found:", filePath)
    process.exit(1)
  }

  return fs.readFileSync(filePath, { encoding: "utf-8" })
}

async function getDocsFromTxt(
  filePath: string,
  name: string
): Promise<Document[]> {
  const content = readFileContent(filePath)
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  })
  return splitter.createDocuments([content], [{ name }])
}

async function getDocsFromPdf(
  filePath: string,
  name: string
): Promise<Document[]> {
  const buffer = fs.readFileSync(filePath)
  const pdfData = await pdfParse(buffer)
  const content = pdfData.text
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  })
  return splitter.createDocuments([content], [{ name }])
}

async function getDocsFromFile(
  filePath: string,
  name: string
): Promise<Document[]> {
  const fileExtension = path.extname(filePath)
  switch (fileExtension) {
    case ".txt":
      return getDocsFromTxt(filePath, name)
    case ".pdf":
      return getDocsFromPdf(filePath, name)
    default:
      throw new Error("Unsupported file extension")
  }
}

async function storeDocuments(
  docs: Document[],
  pineconeIndex: any,
  namespace: string
) {
  await PineconeStore.fromDocuments(docs, new OpenAIEmbeddings(), {
    pineconeIndex,
    namespace,
  })
}

async function main() {
  // Check if a file path argument is provided
  if (process.argv.length < 3) {
    console.error("Please provide a path to a text file.")
    process.exit(1)
  }

  if (process.argv.length < 4) {
    console.error("Please provide a name for the game.")
    process.exit(1)
  }

  // Normalize the provided file path
  const filePath = path.resolve(process.argv[2])

  // Check if a name argument is provided
  const name = process.argv[3]

  const pineconeIndex = await initializeIndex()
  const docs = await getDocsFromFile(filePath, name)
  await storeDocuments(docs, pineconeIndex, name)
  console.log("Rules files added to Pinecone index")
}

main()
