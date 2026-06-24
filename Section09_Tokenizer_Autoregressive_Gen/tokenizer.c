#include "tokenizer.h"

// Hardcoded tiny vocab lookup table for demo
static const char* vocab_table[MAX_TOKEN_VOCAB] = {
    "", "<bos>", "<eos>", "hello", "world", "how", "are", "you",
    "i", "am", "fine", "thank", "you", "what", "is", "llama"
};

/**
 * Simple naive tokenizer: split by space, map word to index
 */
u32 text_to_tokens(const char* text, u32* tokens, u32 max_tokens)
{
    u32 token_cnt = 0;
    char buf[64];
    u32 buf_idx = 0;

    // Prepend BOS token
    if (token_cnt < max_tokens)
        tokens[token_cnt++] = TOKEN_BOS;

    for (u32 i = 0; text[i] != '\0'; i++)
    {
        if (text[i] == ' ' || text[i] == '\n')
        {
            if (buf_idx == 0) continue;
            buf[buf_idx] = '\0';
            buf_idx = 0;

            // Lookup word in vocab
            u32 tid = 0;
            for (; tid < MAX_TOKEN_VOCAB; tid++)
            {
                if (strcmp(buf, vocab_table[tid]) == 0)
                    break;
            }
            if (tid >= MAX_TOKEN_VOCAB) tid = 0;
            if (token_cnt < max_tokens)
                tokens[token_cnt++] = tid;
        }
        else
        {
            if (buf_idx < 63)
                buf[buf_idx++] = text[i];
        }
    }

    // Process remaining word at end of string
    if (buf_idx > 0)
    {
        buf[buf_idx] = '\0';
        u32 tid = 0;
        for (; tid < MAX_TOKEN_VOCAB; tid++)
        {
            if (strcmp(buf, vocab_table[tid]) == 0)
                break;
        }
        if (tid >= MAX_TOKEN_VOCAB) tid = 0;
        if (token_cnt < max_tokens)
            tokens[token_cnt++] = tid;
    }
    return token_cnt;
}

/**
 * Convert token ID back to text word
 */
void token_to_text(u32 token, char* out_str, u32 str_len)
{
    memset(out_str, 0, str_len);
    if (token >= MAX_TOKEN_VOCAB)
        return;
    strncpy(out_str, vocab_table[token], str_len - 1);
}