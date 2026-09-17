package com.uvgraph.b2c.controller;

import java.util.Map;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * 最小 ChatModel 示例：直接调用通义千问（DashScope）。
 */
@RestController
@RequestMapping("/api/chat")
public class ChatController {

    private final ChatClient chatClient;

    public ChatController(ChatClient.Builder chatClientBuilder) {
        this.chatClient = chatClientBuilder.build();
    }

    @GetMapping
    public Map<String, String> chat(@RequestParam(defaultValue = "你好") String message) {
        String reply = chatClient.prompt()
                .user(message)
                .call()
                .content();
        return Map.of("reply", reply);
    }

    @PostMapping
    public Map<String, String> chat(@RequestBody ChatRequest request) {
        String reply = chatClient.prompt()
                .user(request.message())
                .call()
                .content();
        return Map.of("reply", reply);
    }

    public record ChatRequest(String message) {
    }
}
