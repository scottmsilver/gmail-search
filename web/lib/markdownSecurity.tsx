import React from "react";
import ReactMarkdown, { type Components, type Options } from "react-markdown";

// Model and email text is untrusted. Rendering an image URL can disclose
// content embedded in its path/query before the reader chooses to navigate.
// Render descriptions only, even for relative URLs targeting our own APIs.
export const passiveMarkdownComponents: Components = {
  img: ({ alt }) => <span>{alt ? `[Image: ${alt}]` : "[Image omitted]"}</span>,
};

// Use the same policy for answers and expanded reasoning. Callers can customize
// citations, but cannot accidentally restore automatic image fetches or HTML.
export const PassiveMarkdown = ({ components, ...props }: Pick<Options,
  "children" | "components" | "remarkPlugins" | "urlTransform"
>) => (
  <ReactMarkdown {...props} skipHtml components={{ ...components, ...passiveMarkdownComponents }} />
);
