"use client";

import { Suspense } from "react";

import { SearchView } from "@/components/SearchView";
import { ThemeEffect } from "@/components/ThemeEffect";

export default function SearchPage() {
  return (
    <Suspense fallback={null}>
      <ThemeEffect />
      <SearchView />
    </Suspense>
  );
}
