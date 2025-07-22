interface PromptSuggestionsProps {
  label: string;
  suggestions: string[];
  onSuggestionClick?: (suggestion: string) => void;
  append?: (message: { role: "user"; content: string }) => void;
}

export function PromptSuggestions({
  label,
  suggestions,
  onSuggestionClick,
  append,
}: PromptSuggestionsProps) {
  return (
    <div className="space-y-6">
      <h2 className="text-center text-2xl font-bold">{label}</h2>
      <div className="flex gap-6 text-sm">
        {suggestions.map((suggestion) => (
          <button
            key={suggestion}
            onClick={() =>
              onSuggestionClick
                ? onSuggestionClick(suggestion)
                : append && append({ role: "user", content: suggestion })
            }
            className="h-max flex-1 rounded-xl border bg-background p-4 hover:bg-muted"
          >
            <p>{suggestion}</p>
          </button>
        ))}
      </div>
    </div>
  );
}
