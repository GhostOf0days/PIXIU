# def process_text(entity_string, text):
#     # Initialize
#     entity_list = [(", ".join(val.split(", ")[:-1]), val.split(", ")[-1]) for val in entity_string.split("\n")]
#     text_words = text.split()
#     labels = ['O'] * len(text_words)
#     # text_lower = text.lower()
#     text_lower = text

#     # Create a list to store the start index of each word
#     word_indices = [0]
#     for word in text_words[:-1]:
#         word_indices.append(word_indices[-1] + len(word) + 1)

#     # Iterate over the entity list
#     print (entity_list)
#     for entity, entity_type in entity_list:
#         entity_words = entity.split()
#         entity_lower = entity

#         # Find start and end index of each occurrence of the entity in the text
#         start = 0
#         while True:
#             start = text_lower.find(entity_lower, start)
#             if not entity or start == -1: break  # No more occurrence
#             end = start + len(entity) - 1

#             # Find the words included in this occurrence
#             try:
#                 start_word = next(i for i, ind in enumerate(word_indices) if ind >= start)
#                 end_word = next(i for i, ind in enumerate(word_indices) if ind > end)

#                 # Label the words
#                 labels[start_word] = 'B-' + entity_type
#                 for i in range(start_word+1, end_word):
#                     labels[i] = 'I-' + entity_type

#                 # Move to the next character after the occurrence
#             except Exception:
#                 pass
#             start = end + 1

#     return labels

def process_text(entity_string, text):
    print(f"\nDEBUG process_text: Entity string: {entity_string}")
    print(f"DEBUG process_text: Text: {text}")
    
    # Initialize
    try:
        # Split the entity string into lines and extract entity-type pairs
        entity_list = [(", ".join(val.split(", ")[:-1]), val.split(", ")[-1]) for val in entity_string.split("\n") if val.strip()]
        print(f"DEBUG process_text: Parsed entity list: {entity_list}")
    except Exception as e:
        print(f"DEBUG process_text: Error parsing entity string: {e}")
        entity_list = []
    
    text_words = text.split()
    labels = ['O'] * len(text_words)
    text_lower = text
    
    # Create a list to store the start index of each word
    word_indices = [0]
    for word in text_words[:-1]:
        word_indices.append(word_indices[-1] + len(word) + 1)
    
    print(f"DEBUG process_text: Word indices: {word_indices}")
    
    # Iterate over the entity list
    for entity, entity_type in entity_list:
        print(f"DEBUG process_text: Processing entity: '{entity}', type: '{entity_type}'")
        entity_words = entity.split()
        entity_lower = entity
        
        # Find start and end index of each occurrence of the entity in the text
        start = 0
        found = False
        while True:
            start = text_lower.find(entity_lower, start)
            if not entity or start == -1: 
                break  # No more occurrence
            
            found = True
            end = start + len(entity) - 1
            print(f"DEBUG process_text: Found '{entity}' at position {start}-{end}")
            
            # Find the words included in this occurrence
            try:
                start_word = next(i for i, ind in enumerate(word_indices) if ind >= start)
                end_word = next(i for i, ind in enumerate(word_indices) if ind > end)
                
                print(f"DEBUG process_text: Labeling words from {start_word} to {end_word}")
                
                # Label the words
                labels[start_word] = 'B-' + entity_type
                for i in range(start_word+1, end_word):
                    labels[i] = 'I-' + entity_type
                
            except Exception as e:
                print(f"DEBUG process_text: Error labeling entity: {e}")
                pass
            
            start = end + 1
        
        if not found:
            print(f"DEBUG process_text: Entity '{entity}' not found in text")
    
    print(f"DEBUG process_text: Final labels: {labels}")
    return labels