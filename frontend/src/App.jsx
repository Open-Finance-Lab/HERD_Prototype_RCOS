import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./assets/Navbar";
import Footer from "./assets/footer";
import Main from "./assets/Main";
import ContentCard from "./assets/ContentCard";
import About from "./pages/About";
import "./App.css";

function App() {
  const text1 =
    "With AI becoming an integral part of everyday applications, ensuring its accessibility to everyone is more important than ever. The Heterogeneous Expert with Routing Decisions model is designed to provide a highly accurate, open-source AI solution that is free for anyone to use. By making this model openly available, we aim to foster community collaboration, enabling continuous improvements and advancements that benefit all.";
  const text2 = 'Just click the "Chat" link above and type in your prompt to get started!';

  return (
    <Router>
      <Navbar />

      <Routes>
        <Route
          path="/"
          element={
            <Main>
              <div className="title">
                <h1>Heterogeneous Experts with Routing Decisions</h1>
              </div>

              <div className="content">
                <ContentCard
                  imageSrc="/placeHolderImg.png"
                  flip="no"
                  text="Purpose"
                  contentText={text1}
                />
                <ContentCard
                  imageSrc="/placeHolderImg.png"
                  flip="yes"
                  text="How to Use"
                  contentText={text2}
                />
              </div>
            </Main>
          }
        />
        <Route path="/about" element={<About />} />
        {/* <Route path="/chat" element={<Chat />} />
        <Route path="/signin" element={<SignIn />} /> */}
      </Routes>

      <Footer />
    </Router>
  );
}

export default App;
