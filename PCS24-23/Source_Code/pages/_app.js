import "../styles/globals.css";

//INTERNAL IMPORT
import { VotingProvider } from "../context/Voter";
import NavBar from "../components/NavBar/NavBar";
import { SpeedInsights } from "@vercel/speed-insights/next"

const MyApp = ({ Component, pageProps }) => (
  <VotingProvider>
    <div>
      <NavBar />
      <div>
        <Component {...pageProps} />
      </div>
    </div>
    <SpeedInsights />
  </VotingProvider>
);

export default MyApp;
