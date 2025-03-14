import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import Navbar from './assets/Navbar'
import Footer from './assets/footer'
import Main from './assets/Main'
import ContentCard from './assets/ContentCard'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <Navbar></Navbar>

      <Main>
        <div className="title">
          <h1>Heterogenous Experts with Routing Decisions</h1>
        </div>

        <div className="content">
          <ContentCard imageSrc="/placeHolderImg.png" flip="no" text="Hello, World!" />

          <ContentCard imageSrc="/placeHolderImg.png" flip="yes" text="Bye bye, World!" />
        </div>

      </Main>
      
      <Footer></Footer>
    </>
  )
}

export default App
  