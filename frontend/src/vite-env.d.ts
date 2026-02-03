/// <reference types="vite/client" />

// WGSL shader modules
declare module '*.wgsl?raw' {
  const content: string;
  export default content;
}
