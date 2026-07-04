import HeroSection from "@/components/hero/HeroSection";
import KineticText from "@/components/sections/KineticText";
import ParallaxGallery from "@/components/sections/ParallaxGallery";
import SplitSection from "@/components/sections/SplitSection";
import StaggeredGrid from "@/components/sections/StaggeredGrid";

export default function Home() {
  return (
    <main>
      <HeroSection />
      <KineticText />
      <ParallaxGallery />
      <SplitSection />
      <StaggeredGrid />
    </main>
  );
}
