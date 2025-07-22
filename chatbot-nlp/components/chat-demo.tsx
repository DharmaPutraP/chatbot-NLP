"use client";

import { useChat } from "ai/react";
import React from "react";
import { Chat } from "@/components/ui/chat";
import axios from "axios";

export function ChatDemo() {
  // Fungsi untuk handle klik suggestion
  const handleSuggestionClick = async (suggestion: string) => {
    append({ role: "user", content: suggestion });
    setInput("");
    try {
      const response = await axios.post(
        `${process.env.API_URL}/chat`,
        { message: suggestion },
        { headers: { "Content-Type": "application/json" } }
      );
      if (response.data && response.data.response) {
        append({ role: "assistant", content: response.data.response });
      }
    } catch (error) {
      append({ role: "assistant", content: "Maaf, terjadi kesalahan server." });
    }
  };
  const { messages, input, handleInputChange, append, status, stop, setInput } =
    useChat();

  const isLoading = status === "submitted" || status === "streaming";

  const [suggestions, setSuggestions] = React.useState<string[]>([]);

  React.useEffect(() => {
    axios({
      method: "get",
      url: `${process.env.API_URL}/suggestions`,
      responseType: "json",
    })
      .then(function (response) {
        if (response.data && response.data.suggestions) {
          setSuggestions(response.data.suggestions);
        }
      })
      .catch(function (error) {
        console.error("Gagal memuat data:", error);
      });
  }, []);

  const handleSubmit = async (event?: { preventDefault?: () => void }) => {
    if (event && event.preventDefault) event.preventDefault();
    if (!input.trim()) return;
    append({ role: "user", content: input });
    setInput("");
    try {
      const response = await axios.post(
        `${process.env.API_URL}/chat`,
        { message: input },
        { headers: { "Content-Type": "application/json" } }
      );
      if (response.data && response.data.response) {
        append({ role: "assistant", content: response.data.response });
      }
    } catch (error) {
      append({ role: "assistant", content: "Maaf, terjadi kesalahan server." });
    }
  };

  return (
    <Chat
      messages={messages}
      input={input}
      handleInputChange={handleInputChange}
      handleSubmit={handleSubmit}
      isGenerating={isLoading}
      stop={stop}
      append={append}
      suggestions={suggestions}
      handleSuggestionClick={handleSuggestionClick}
    />
  );
}
