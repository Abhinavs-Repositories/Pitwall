import HeroSection from "@/components/hero/HeroSection";
import KineticText from "@/components/sections/KineticText";
import ParallaxGallery from "@/components/sections/ParallaxGallery";
import SplitSection from "@/components/sections/SplitSection";
import HorizontalCarousel from "@/components/sections/HorizontalCarousel";
import Footer from "@/components/sections/Footer";

export default function Home() {
  return (
    <main>
      <HeroSection />
      <KineticText />
      <ParallaxGallery />
      <SplitSection />
      <HorizontalCarousel />
      <Footer />
    </main>
  );
}
